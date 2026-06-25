# LMHC-Forge — Results Record

This file maps each replaced paper claim to the measured artifact result.
Every number in this file comes from the RTX 4050 Laptop GPU (Ada Lovelace,
sm_89, WDDM/WSL2 passthrough) unless stated otherwise.

---

## Claim-Mapping Summary

| Paper claim | Paper's original (estimated) value | Measured artifact result | File / Figure |
|---|---|---|---|
| **§6.2** — Recovery rate: fraction of shared-memory words readable by a co-resident listener kernel | ">90% of words" (estimated) | **EI = 1.000 → 0.000** (N=200 trials, 200/200 same-SM): 100% without scrub; 0% with LMHC hardened scrub | `results/leakage_demo.json`; `figures/fig3_exposure_index` |
| **Table I / §6.1** — Scrub overhead for three kernel classes (reduction, GEMM, attention) | "1–5% estimated" (all six cells placeholder) | **Below timing noise floor** on RTX 4050: σ upper bounds < 1.08 µs (GEMM), < 1.26 µs (reduction), < 1.77 µs (attention); isolation trend +0.17 µs for attention (positive, unresolved at 1σ) | `results/benchmark.json`; `figures/fig4_benchmark` |
| **§4.4 / Theorem 1** — DSE hazard: naive zeroing eliminated by compiler; volatile+fence survives | Elimination claimed for naive scrub; survival claimed for volatile+fence | **Ordering differential** (not per-store elimination) on ptxas 12.8 and 13.3 at -O3 sm_89: STS counts equal (58 each); hardened uniquely emits `MEMBAR.SC.CTA` + extra `BAR.SYNC`; pure-dead allocation-granularity DSE confirmed (naive → 0 STS, volatile → 29 STS where buffer never read) | `results/dse_differential.json`; `sass_dumps/dse_test.sass`; `figures/fig2_dse_differential` |
| **§5.4** — Verifier independence: attester re-derives scrub presence from binary without trusting witness fields | "Independent re-derivation" (asserted, not demonstrated) | **T1–T6 tamper-test ladder**: T1–T5 correctly FAIL (hash mismatch, missing scrub in SASS, unaccounted kernel, missing source hash); T6 PASS (known limitation — generate-from-scratch requires binary-level taint to close) | `src/attester.py`; `src/tamper_tests.py`; `figures/fig5_tamper_panel` |

*Hardware caveat: all measurements on RTX 4050 Laptop GPU (Ada Lovelace, sm_89, WDDM/WSL2 passthrough). Relative overhead generalises qualitatively; absolute µs values do not transfer to H100 or other targets.*

---

## Phase 0 — Residue Detector [COMPLETE — 2026-06-25]

**Paper claim**: §6 states residue windows exist for cross-kernel shared-memory
reuse. Phase 0 measures the baseline "before LMHC" recovery rate.

**Measured result (N=1000 trials):**

| Run                              | Non-zero words | Pattern 0xCAFEBABE | Residue rate |
|----------------------------------|---------------:|-------------------:|-------------:|
| Kernel A then Kernel B (same SM) | 4 096 000 / 4 096 000 | 4 096 000 | 1.000000 |
| Control: Kernel B only (no A)    |             0 / 4 096 000 |         0 | 0.000000 |

**Interpretation**: 0xCAFEBABE appears if and only if Kernel A ran first in
the same process. The control (Kernel B launched cold, Kernel A never run)
reads zero in every word across all 1000 trials. This rules out
self-contamination: the residue is cross-launch, not self-read.

**Mandatory framing (Module 6, verbatim)**:
> Residue is fully observable when the subsequent kernel is co-resident on
> the same SM — co-residency is forced in this experiment to demonstrate the
> window. This is in-process cross-kernel shared-memory persistence, NOT
> cross-tenant LeftoverLocals on NVIDIA.

Diff-SM was 0/1000 because `__launch_bounds__(32,1)` + single-block launches
force back-to-back scheduling on the same SM. Co-residency is forced, not a
natural statistical occurrence. The experiment demonstrates THAT the window
exists; it does not characterise HOW OFTEN it arises in production pipelines.

**Result files**: `results/phase0_residue.json`, `results/phase0_control.json`

---

## Module 1 — Discovery Engine [COMPLETE — 2026-06-25]

**Function**: Identify `__shared__` buffers that require LMHC scrubbing.

**Input**: `src/test_kernels.cu` (3 kernels: attention_fwd, reduction_sum, gemm_tile)

### Conservative pass (flag every `__shared__` buffer)

| Kernel        | Buffer        | Type  | Sensitivity   |
|---------------|---------------|-------|---------------|
| attention_fwd | smem_qk       | float | conservative  |
| attention_fwd | smem_softmax  | float | conservative  |
| reduction_sum | smem_partial  | float | conservative  |
| gemm_tile     | a_tile        | float | conservative  |
| gemm_tile     | b_tile        | float | conservative  |

**Total: 5 of 5 buffers flagged.**

### Taint pass (forward-propagate from `__confidential__` parameters)

| Kernel        | Buffer        | Type  | Sensitivity      |
|---------------|---------------|-------|------------------|
| attention_fwd | smem_qk       | float | **confidential** |
| attention_fwd | smem_softmax  | float | non-confidential |
| reduction_sum | smem_partial  | float | non-confidential |
| gemm_tile     | a_tile        | float | **confidential** |
| gemm_tile     | b_tile        | float | non-confidential |

**Total: 2 of 5 buffers flagged (smem_qk, a_tile).**

Taint propagation is syntactic (intra-procedural, line-level), not semantic.
See SPEC.md LIMITATIONS for scope.

**Result files**: `results/discovery_conservative.json`, `results/discovery_taint.json`

---

## Module 2 — LMHC Rewriter [COMPLETE — 2026-06-25]

**Function**: Insert volatile+fence scrub blocks before each kernel exit for
every flagged `__shared__` buffer.

**Scope**: Single-exit kernels only. All three test kernels qualify (0 explicit
returns). Multi-exit kernels are rejected with an explicit error — see SPEC.md
LIMITATIONS.

**Taint-only run results**:

| Kernel        | Status    | Scrubbed buffers |
|---------------|-----------|------------------|
| attention_fwd | rewritten | smem_qk          |
| reduction_sum | rewritten | (none — no confidential buffers) |
| gemm_tile     | rewritten | a_tile           |

Scrub pattern inserted before each kernel's closing `}`:
```cuda
/* BEGIN LMHC scrub — BUFNAME (NELEMS x TYPE) */
{
    volatile TYPE *__lmhc_p = (volatile TYPE *)BUFNAME;
    for (int __lmhc_i = (int)threadIdx.x;
         __lmhc_i < (int)(NELEMS);
         __lmhc_i += (int)blockDim.x)
        __lmhc_p[__lmhc_i] = (TYPE)0;
}
__threadfence_block();
__syncthreads();
/* END LMHC scrub — BUFNAME */
```

**Output**: `src/test_kernels_lmhc.cu`

---

## Module 3 — DSE Differential [COMPLETE — 2026-06-25]

**Paper claim** (Theorem 1 / §4.4): A naive `for(...) buf[i]=0` scrub is
subject to dead-store elimination by an optimizing compiler, leaving sensitive
data in shared memory after kernel exit. The volatile+fence version cannot be
eliminated.

**Hardware**: RTX 4050 Laptop GPU (Ada Lovelace, sm_89)  
**Compilers tested**: ptxas 12.8 (V12.8.93) and ptxas 13.3 (V13.3.33)  
**Compile flags**: `-arch=sm_89 -O3`  
**Analysis tool**: `cuobjdump -sass` (both 12.8 and 13.3)

### Finding (a): ptxas performs allocation-granularity DSE only

ptxas eliminates shared-memory stores ONLY when the entire `__shared__`
allocation has no reads anywhere in the kernel. In that case all stores to
the buffer are dropped and ptxas emits warning `#550-D: variable "buf" was
set but never used`. This is whole-allocation elimination, not per-store
analysis.

| Kernel               | ptxas 12.8 STS-data | ptxas 12.8 STS-RZ | ptxas 13.3 STS-data | ptxas 13.3 STS-RZ | Total (12.8 / 13.3) |
|----------------------|--------------------:|------------------:|--------------------:|------------------:|--------------------:|
| nosync_naive         |                  29 |                29 |                  29 |                29 |          58 / 58    |
| nosync_volatile      |                  29 |                29 |                  29 |                29 |          58 / 58    |
| pure_dead_naive      |                   0 |                 0 |                   0 |                 0 |           **0 / 0** |
| pure_dead_volatile   |                   0 |                29 |                   0 |                29 |          29 / 29    |

`STS-data` = Phase 1 computation stores (registers R6, R7, … — live, retained).  
`STS-RZ`   = scrub zero-stores (`STS [addr], RZ` — dead candidates).

`pure_dead_naive`: buffer written in the scrub loop but **never read** anywhere
in the kernel. ptxas eliminates the entire allocation → 0 STS, warning `#550-D`.

### Finding (b): the LMHC fill→read→scrub pattern is retained on both versions

`nosync_naive` represents the actual LMHC threat scenario: buffer filled with
sensitive data (Phase 1), read to produce output (Phase 2, no `__syncthreads__`
before the scrub), then zeroed (Phase 3). The 29 scrub zero-stores survive on
BOTH ptxas 12.8 and ptxas 13.3 — byte-identical SASS. Removing the
`__syncthreads()` between the live read and the scrub made no difference.

ptxas's shared-memory DSE engine operates at whole-allocation granularity
("this buffer has no reads at all"), not at per-store granularity ("this
specific store has no subsequent reads"). The fill→read→scrub pattern falls
entirely outside the scope of elimination on either tested version.

### Finding (c): volatile is load-bearing where elimination is active

`pure_dead_volatile` vs `pure_dead_naive`:
- naive (0 STS): ptxas drops the entire allocation.
- volatile (29 STS): volatile forces the stores to survive even when the
  buffer is never read. The scrub zero-stores (all `STS [addr], RZ`) are
  present and counted.

This proves that the `volatile` qualifier does exactly what Theorem 1 requires
— it makes stores un-eliminable — at the granularity where elimination is
actually implemented. On BOTH ptxas 12.8 and 13.3, `pure_dead_volatile` → 29
STS where `pure_dead_naive` → 0 STS.

### Finding (d): the empirical differential on the fill→read→scrub pattern is ordering

For the kernels that model the actual LMHC scrub (from `src/dse_test.cu`):

| Kernel           | STS | MEMBAR (threadfence_block) | BAR.SYNC (syncthreads) |
|------------------|----:|---------------------------:|-----------------------:|
| scrub_naive      |  58 |                          0 |                      2 |
| scrub_hardened   |  58 |                          1 |                      3 |
| scrub_ptx_fallback | 58 |                          1 |                      3 |

The scrub zero-stores (29 STS with RZ) survive in ALL THREE versions on
ptxas 12.8. The machine-checked differential is in ordering instructions:
`scrub_hardened` uniquely emits `MEMBAR.SC.CTA` (`__threadfence_block()`)
and an additional `BAR.SYNC` (`__syncthreads()`). `scrub_ptx_fallback` is
identical to hardened, confirming the PTX inline-asm fallback produces the
same guarantee.

**Theorem 1 status**: Theorem 1 is a *permission-level* argument under the
C++ abstract machine: a conforming optimizer is **permitted** to eliminate
plain `buf[i]=0` stores because they have no subsequent reads. The volatile
qualifier makes stores an *observable side-effect* that no conforming
implementation may remove.

On ptxas 12.8 and 13.3 at -O3, that permitted elimination does not fire for
the fill→read→scrub pattern. The empirical contribution of the hardened scrub
on this toolchain is the **ordering guarantee** (MEMBAR.SC.CTA ensures all
scrub stores complete before kernel exit under the C++ memory model), not
store presence. The naive version's stores happen to be present empirically
but carry no such guarantee.

**Do not relabel the ordering differential as elimination.** These are
distinct claims at distinct levels: one is about what the compiler is
permitted to do; the other is about what it actually did on a specific
version.

**Result files**: `results/dse_differential.json`, `sass_dumps/dse_test.sass`

---

## Module 4 — Witness Generator [COMPLETE — 2026-06-25]

**Function**: Compile source to device binary, compute SHA-384(cubin || policy),
emit `witness.json`.

**Input**: `src/test_kernels.cu` (via Module 2 rewriter — taint pass, 2 kernels rewritten)  
**Compiler**: nvcc 12.8, `--cubin -arch=sm_89 -O3 -ccbin /usr/bin/gcc-14`

**Hash construction**:
```
kernel_hash = SHA-384(cubin_bytes || json.dumps(policy_doc, sort_keys=True, separators=(',',':')))
```
where `policy_doc` contains: `all_exits_protected`, `arch`, `compile_flags`,
`policy_version`, `scrub_sites`, `sensitive_regions`.

**Measured outputs**:

| Field               | Value |
|---------------------|-------|
| `policy_version`    | `lmhc-v1.0` |
| `arch`              | `sm_89` |
| `cubin_size_bytes`  | 10 656 |
| `all_exits_protected` | `true` |
| `sensitive_regions` | `attention_fwd.smem_qk`, `gemm_tile.a_tile` |
| `scrub_sites`       | `gemm_tile.a_tile`, `attention_fwd.smem_qk` |
| `kernel_hash`       | `b916d3e73e5514759e65bd7b13e6ace55e4c2a2f0a06c723be5ed734697c416de5c984bdd4cffd08341563aecfcb83f3` |

**Result files**: `results/witness.json`, `results/test_kernels.cubin`

---

## Module 5 — Attestation Verifier [COMPLETE — 2026-06-25]

**Function**: Re-derive scrub presence from `.cubin` alone via `cuobjdump -sass`;
independently recompute SHA-384; PASS only if all three checks pass.

**Independence requirement**: The verifier does NOT trust `all_exits_protected`,
`scrub_sites`, or `sensitive_regions` as ground truth. `sensitive_regions` is
the claim of what to check; each listed kernel is verified structurally from SASS.
Non-nominated kernels are checked for confidential taint via Module 1.

### Three checks

**CHECK 1 — Hash integrity**: Reconstruct `policy_doc` from witness fields (same
key set / serialisation as Module 4), compute SHA-384(cubin || policy_json),
compare to `kernel_hash`. Detects modification to the binary or any hashed policy
field.

**CHECK 2 — Structural scrub verification**: For every kernel in `sensitive_regions`,
disassemble with `cuobjdump -sass` and verify `STS [addr], RZ → MEMBAR.SC.CTA →
BAR.SYNC` ordering before EXIT. Derived from SASS only — does not consult
`scrub_sites`.

**CHECK 3 — Kernel inventory completeness**: Enumerate all kernels in the cubin
(from cuobjdump). For every kernel NOT in `sensitive_regions`, run Module 1's
taint pass on the source named in the witness. If any non-nominated kernel has a
confidential-derived `__shared__` buffer → FAIL with "unaccounted sensitive kernel:
`<name>`". This makes the sensitive-region nomination a CHECKED claim, not a
trusted one.

### Legitimate cubin — T3 baseline

```
[smem_qk] attention_fwd: PASS
          STS.RZ@207 < MEMBAR@217 < BAR.SYNC@219 < EXIT@221  [1 zero-store(s)]
[a_tile]  gemm_tile:     PASS
          STS.RZ@193 < MEMBAR@203 < BAR.SYNC@205 < EXIT@207  [1 zero-store(s)]
  reduction_sum: OK -- no confidential __shared__ buffers (taint-verified)
VERDICT: PASS
```

### Tamper tests — second run after CHECK 3 added (no edits to attester.py between runs)

The stripped cubin was produced by removing the LMHC scrub block from `gemm_tile`
(volatile loop + `__threadfence_block()` + `__syncthreads()`) and recompiling.
Result: 10 400 bytes (vs 10 656), no `STS [addr], RZ` in `gemm_tile`.

| Test | Setup | CHECK 1 | CHECK 2 | CHECK 3 | VERDICT |
|------|-------|---------|---------|---------|---------|
| **T1** | stripped cubin + original witness | FAIL (hash mismatch) | gemm FAIL | reduction_sum OK | **FAIL** |
| **T2** | stripped cubin + regenerated hash; gemm still in `sensitive_regions` | PASS | gemm FAIL | reduction_sum OK | **FAIL** |
| **T3** | original cubin + original witness | PASS | both kernels PASS | reduction_sum OK | **PASS** |
| **T4** | stripped cubin + regenerated hash; gemm DROPPED from `sensitive_regions` | PASS | attention_fwd PASS | gemm FAIL (unaccounted) | **FAIL** |

**T1 exact FAIL reasons**:
```
✗ hash_integrity: hash mismatch: recomputed=cadb7d9bd3cfd009… claimed=b916d3e73e551475…
✗ struct_gemm_tile_a_tile: gemm_tile: FAIL — no STS [addr], RZ before EXIT
  (scrub zero-stores absent or eliminated)
```

**T2 exact FAIL reason** (CHECK 1 passes; CHECK 2 fires from SASS alone):
```
✗ struct_gemm_tile_a_tile: gemm_tile: FAIL — no STS [addr], RZ before EXIT
  (scrub zero-stores absent or eliminated)
```

**T4 exact FAIL reason** (CHECK 1 and CHECK 2 both pass; CHECK 3 catches the attack):
```
✗ kernel_inventory: FAIL -- unaccounted sensitive kernel: gemm_tile
  (tainted __shared__ buffer(s): a_tile)
```

T4 returns FAIL. Without CHECK 3, it returned PASS silently.

### T5 — Source de-annotation attack (trust boundary test for CHECK 3 + CHECK 1 fix)

**T5 setup**: stripped cubin + de-annotated source (gemm_tile's `__confidential__`
removed so taint pass finds nothing) + regenerated witness pointing to de-annotated
source, with gemm_tile dropped from `sensitive_regions`. Witness generated with the
**old-format kernel_hash** (no `source_file_hash` in policy_doc).

**T5 before fix** (CHECK 3 present, but `source_file_hash` not yet in policy_doc):

CHECK 3 reads `source_file = /tmp/test_kernels_deannotated.cu` (the attacker's
de-annotated file). Taint finds `gemm_tile.a_tile` is non-confidential → CHECK 3
passes. CHECK 1 passes (hash matches; source file bytes are not in the hash formula).
**VERDICT: PASS (hole demonstrated — de-annotation defeats CHECK 3 and CHECK 1
simultaneously).**

**Fix applied**: `source_file_hash = SHA-384(source_file_bytes)` added to
`policy_doc` in Module 4 (`witness_gen.py`). `attester.py` CHECK 1 now
independently re-hashes the bytes at `witness.source_file` and includes the result
in the policy_doc reconstruction. This mirrors CHECK 2's independence: SASS is
re-derived from the binary, not trusted from witness fields.

**T5 after fix** (same old-format T5 witness, updated verifier):

```
── CHECK 1: Hash integrity ──────────────────────────────────
  source_file      : /tmp/test_kernels_deannotated.cu
  source_file_hash : ac7885d0d9d05f1bb82d20cf83730885d5795e15…  (independently computed)
  recomputed : 033852118da18f88db561cb132c0c2e81d179317…
  claimed    : 8a499d241a5e0a22e8433fa1b89e4c1dfaad347b…
  → MISMATCH — cubin or policy fields have been tampered with
VERDICT: FAIL
✗ hash_integrity: hash mismatch: recomputed=033852118da18f88… claimed=8a499d241a5e0a22…
```

CHECK 1 now fails because the T5 witness's `kernel_hash` was computed without
`source_file_hash` in the policy_doc, but the updated verifier adds it (regardless
of which source file is named in the witness). The two hash formulas cannot match.

### T6 — Generate-from-scratch (KNOWN LIMITATION, not closed)

**T6 setup**: Adversary runs the updated `witness_gen.py` on a de-annotated original
source (gemm_tile's `__confidential__` removed) and lets the full Module 4 pipeline
generate a fresh, internally consistent witness.

What Module 4 does on the de-annotated source:
- Taint pass: finds only `attention_fwd.smem_qk` (gemm_tile.a_tile no longer tainted)
- Rewriter: inserts scrub only for `attention_fwd` — gemm_tile receives none
- Compile: cubin has no `STS [addr], RZ` in gemm_tile (10 400 bytes, same as stripped)
- `source_file_hash = SHA-384(de-annotated source bytes)` = `298899e326e8469eb9ff…`
- `kernel_hash = SHA-384(T6_cubin || policy_with_deannotated_source_hash)` = `874786ce1d547a84…`

**T6 verifier output**:

```
── CHECK 1: Hash integrity ──────────────────────────────────
  source_file      : /tmp/test_kernels_t6_input.cu
  source_file_hash : 298899e326e8469eb9ff29b9baf82b12…  (independently computed)
  recomputed : 874786ce1d547a8469fe0f82545aa5a7…
  claimed    : 874786ce1d547a8469fe0f82545aa5a7…
  → MATCH

── CHECK 2: Structural scrub verification (SASS) ───────────
  [smem_qk] attention_fwd: PASS — STS.RZ@207 < MEMBAR@217 < BAR.SYNC@219 < EXIT@221

── CHECK 3: Kernel inventory completeness ───────────────────
  kernels to inventory : gemm_tile, reduction_sum
  gemm_tile: OK -- no confidential __shared__ buffers
             (taint-verified from /tmp/test_kernels_t6_input.cu)
  reduction_sum: OK -- no confidential __shared__ buffers

── VERDICT ─────────────────────────────────────────────────
  PASS
  ✓ SHA-384(cubin || policy) matches kernel_hash
  ✓ Every nominated kernel contains STS.RZ → MEMBAR.SC.CTA → BAR.SYNC before EXIT
  ✓ Every non-nominated kernel has no confidential __shared__ buffers (taint-verified)
```

**VERDICT: PASS (undetected attack).**

**Why the system cannot close T6 at source level**: The verifier independently
re-hashes `witness.source_file` and re-derives taint from that file. Because the
attacker controls BOTH the file at that path AND the witness hash (regenerated by
the same Module 4 run), every independent computation the verifier performs is
internally consistent with the attacker's de-annotated world. There is no
source-level invariant the verifier can check without an external trusted anchor
(a signed source registry, HSM-sealed annotation, or binary-level taint).

Only binary-level taint re-derivation from SASS — tracing which shared-memory
writes carry data that originated from pointer arguments without de-referencing
through zero — can detect that `gemm_tile.a_tile` holds confidential input data
regardless of what the source file says about it.

### Full tamper-test matrix (after all fixes)

| Test | Cubin | Source | Witness format | CHECK 1 | CHECK 2 | CHECK 3 | VERDICT |
|------|-------|--------|---------------|---------|---------|---------|---------|
| **T1** | stripped | — | original (hash mismatch) | FAIL | gemm FAIL | reduction_sum OK | **FAIL** |
| **T2** | stripped | — | regenerated, gemm in regions | PASS | gemm FAIL | reduction_sum OK | **FAIL** |
| **T3** | original | — | original | PASS | both PASS | reduction_sum OK | **PASS** |
| **T4** | stripped | original | gemm dropped | PASS | attn PASS | gemm FAIL (tainted) | **FAIL** |
| **T5** | stripped | de-annotated | old-format no source_hash | FAIL (formula) | attn PASS | gemm cleared | **FAIL** |
| **T6** | naturally stripped (M4 on de-annotated) | de-annotated | new-format, regenerated | PASS | attn PASS | gemm cleared | **PASS (KNOWN LIMITATION)** |

### §5.4 / LIMITATIONS: CHECK 1+3 trust boundary (mandatory, per paper honesty requirement)

The source hash fix closes the T5 attack **for witnesses generated with the old format**
(no `source_file_hash` in policy_doc). **One residual trust gap remains, demonstrated
by T6**:

1. **Full re-generation defeats all three checks (T6 — measured).** An adversary
   who controls the witness-generation tool can run the updated `witness_gen.py`
   on a de-annotated source file. The resulting witness is internally consistent:
   `source_file_hash` commits to the de-annotated file; `kernel_hash` commits to
   the cubin compiled from the de-annotated rewrite; `sensitive_regions` honestly
   lists only what taint found. The verifier independently re-derives `source_file_hash`
   from the same de-annotated file and confirms CHECK 1. CHECK 2 passes (attention_fwd
   is protected). CHECK 3 reads the de-annotated source, finds gemm_tile.a_tile
   non-confidential, and clears it. T6 verdict: **PASS** — the source-level attestation
   system cannot distinguish a legitimate witness from an adversarially generated one
   when the adversary controls the source file path named in the witness.

   **Closing requires**: source annotation to be verified independently of any
   file path the witness names — either a trusted source registry (external anchor),
   or binary-level taint re-derivation from SASS (data-flow analysis over SASS
   instructions that identifies which `__shared__` writes carry data derived from
   pointer arguments, independent of source annotations).

2. **Taint is source-annotation-level, not binary-level**. Module 1's taint pass
   is syntactic (regex/intra-procedural) over `__confidential__`-annotated
   parameters. It will miss: confidential data introduced via pointer aliasing,
   function calls not visible in the kernel body, or annotated parameters that were
   compiled away. Binary-level taint from SASS would eliminate the source dependency
   entirely. That remains future work.

Within those boundaries, the six-test suite demonstrates that the attester correctly
classifies T1–T5 and that T6 is the precise boundary of source-level attestation —
a measured result, not a conjectured one. §5.4 of the paper should cite T6 as the
empirical demarcation between what source-level contracts can prove and what requires
binary-level verification.

**Result files**: `src/attester.py`, `src/witness_gen.py`, `src/tamper_tests.py`

---

## LIMITATIONS

See also SPEC.md LIMITATIONS section.

### Toolchain
- All CUDA compilation: WSL2 Kali Linux 2026.1 (glibc 2.42), CUDA 12.8 +
  gcc-14. Module 3 DSE comparison also run with nvcc/ptxas 13.3.33.
- One-time patch applied to `math_functions.h` for glibc 2.42 compatibility
  (see SPEC.md). Original preserved at `math_functions.h.orig`.

### Phase 0 co-residency
- Diff-SM was 0/1000 — co-residency is forced, not natural. The experiment
  demonstrates existence of the window, not its production frequency.

### Module 2 rewriter scope
- Single-exit kernels only. Multi-exit kernels are rejected with an error.

### Module 3 scope
- Tested on sm_89 (Ada Lovelace). Elimination behaviour may differ on other
  architectures or future ptxas versions.
- `nosync_volatile` and `nosync_naive` produce identical SASS on both ptxas
  versions for the fill→read→scrub pattern — confirming volatile adds no
  store-presence benefit there, only ordering guarantees.

---

## Module 6 — Leakage Demo [COMPLETE — 2026-06-25]

**Paper claim** (§6.2): ">90% of words" recoverable from a writer kernel's
shared memory by a subsequent co-resident reader. Module 6 replaces that
estimate with a measured Exposure Index on the RTX 4050.

**Hardware**: RTX 4050 Laptop GPU (Ada Lovelace, sm_89)  
**Buffer**: 4096 × uint32 = 16 KB (matches Phase 0)  
**Sentinel**: 0xCAFEBABE  
**Trials per phase**: 200

**Mandatory framing (SPEC.md §Module 6, verbatim)**:
> Residue is fully observable when the subsequent kernel is co-resident on
> the same SM — co-residency is forced in this experiment to demonstrate the
> window. This is in-process cross-kernel shared-memory persistence, NOT
> cross-tenant LeftoverLocals on NVIDIA.

### Measured Exposure Index

| Phase | Writer | Same-SM trials | Non-zero words | EI |
|-------|--------|---------------:|---------------:|---:|
| A | Without LMHC scrub | 200 / 200 | 819 200 / 819 200 | **1.000000** |
| B | With LMHC hardened scrub | 200 / 200 | 0 / 819 200 | **0.000000** |

**Replaces paper claim**: ">90% recovery" (§6.2) → **100% recovery (EI = 1.0)**
without scrub; **0% recovery (EI = 0.0)** with LMHC hardened scrub. The scrub
reduces the Exposure Index from 1.0 to 0.0 — the residue window is closed.

Co-residency was 200/200 trials because `__launch_bounds__(32,1)` forces back-to-back
single-block launches onto the same SM. Co-residency is forced, not a statistical
rate; the experiment demonstrates existence of the window, not production frequency.

**Result file**: `results/leakage_demo.json`

---

## Module 7 — Benchmark Suite [COMPLETE — 2026-06-25]

**Paper claim** (Table I, §6.1): All six overhead cells (absolute µs and relative %)
for three kernel classes are estimated / placeholder. Module 7 replaces them with
measured values on the RTX 4050.

**Hardware**: RTX 4050 Laptop GPU (Ada Lovelace, sm_89)  
**Timing method**: Interleaved batch measurement — for each rep, 200 baseline launches
followed immediately by 200 scrubbed launches in the same GPU stream with three CUDA
events (start, mid, end). **Per-rep overhead = (scrubbed_half − baseline_half) / 200.**
Median and stddev are computed from the N=50 per-rep overhead values, not from
subtracting medians of separately-collected baseline and scrubbed distributions.
Global warmup: 500 round-robin iterations of all 6 kernels before any timing.

**Scrub-isolation method** (for kernels below the noise floor): An empty kernel
(same shared-memory allocation, one volatile read) is timed against a scrub-only
kernel (same allocation, full LMHC scrub, no computation). Isolates the pure scrub
cost from computation-phase variance. N=80 reps × 1 000 launches/batch.

### Overhead summary

**All three kernel classes show overhead below the timing noise floor** — the
measurement method's ~1–2 µs launch-to-launch scheduling jitter exceeds the
sub-microsecond scrub cost for single-block kernels on the RTX 4050.

Negative medians in the table below are measurement artifacts (variance in which
half of an interleaved rep runs faster); they must NOT be read as "scrub makes
the kernel faster." The correct reading is: overhead is positive but too small to
resolve with CUDA events on this hardware. Results are reported as **upper bounds**.

| Kernel class | Baseline (µs) | Overhead median (µs) | Overhead σ (µs) | Below noise floor? | **Upper bound** | Isolated scrub (µs) | Occupancy |
|---|---|---|---|---|---|---|---|
| Reduction (memory-bound) | 7.91 ± 1.62 | −0.19 | σ = **1.26 µs** | Yes | **< 1.26 µs** | −0.21 ± 1.06 (unresolved) | 50.0% |
| GEMM tile (compute-bound) | 13.93 ± 2.18 | −0.34 | σ = **1.08 µs** | Yes | **< 1.08 µs** | (not run — same conclusion) | 100.0% |
| Attention-style (memory-heavy) | 8.44 ± 3.18 | −1.04 | σ = **1.77 µs** | Yes | **< 1.77 µs** | +0.17 ± 1.23 (positive trend, unresolved) | 20.8% |

### Table I cell mapping (six replaced estimates)

| Table I cell | Paper value | Measured result |
|---|---|---|
| `reduction_overhead_abs_us` | estimated | scrub cost **< 1.26 µs** (noise floor; see §M7 interpretation) |
| `reduction_overhead_pct`    | estimated | **< 16%** of 7.91 µs baseline |
| `gemm_overhead_abs_us`      | estimated | scrub cost **< 1.08 µs** (noise floor) |
| `gemm_overhead_pct`         | estimated | **< 7.8%** of 13.93 µs baseline |
| `attention_overhead_abs_us` | estimated | scrub cost **< 1.77 µs**; isolation gives +0.17 µs (unresolved) |
| `attention_overhead_pct`    | estimated | **< 21%** of 8.44 µs baseline (bound); isolation trend: +2.1% |

### Interpretation

**Measurement limitation**: Single-block kernel CUDA event timing has ~1–2 µs
launch-to-launch scheduling jitter on the RTX 4050. The LMHC scrub adds volatile
stores + `__threadfence_block()` + `__syncthreads()` that complete in < 1 µs based
on shared-memory write latency (measured in isolation as +0.17 µs for attention's
2048-word scrub at 64 threads). This is below the timing resolution of CUDA events
for short single-block kernels.

**What the scrub-isolation runs show**:
- **Reduction** (4 096 words, 256 threads, 16 stores/thread): isolation noise floor
  σ = 1.06 µs; overhead unresolvable. Upper bound: < 1.06 µs.
- **Attention** (2 048 words, 64 threads, 32 stores/thread): isolation median =
  +0.17 µs (positive, expected direction) with σ = 1.23 µs. The positive trend is
  consistent with a small real cost but is not statistically significant at 1σ.
  Upper bound: < 1.23 µs from isolation (< 1.77 µs from interleaved run).

**Why the scrub cost is below the noise floor**: The dominant latency for all three
kernels is either global-memory access (reduction reads 16 KB; attention scatter-gathers
SEQ×HEAD words from global) or CUDA kernel launch overhead (~4–7 µs fixed cost per
launch). The scrub's shared-memory zero-stores are hidden behind these larger latencies.
The LMHC overhead is real and bounded, but it is small enough that CUDA event timing
of single-block kernels cannot isolate it from the fixed launch overhead on this hardware.

**Attention occupancy (sm_89)**: The 8.25 KB shared footprint limits the attention
kernel to 5 active blocks / SM → 5 × 64 / 1 536 = **20.8%** theoretical occupancy.

**Result file**: `results/benchmark.json`
