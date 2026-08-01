# LMHC-Forge — Results Record

This file maps each replaced paper claim to the measured artifact result.
Every number in this file comes from the RTX 4050 Laptop GPU (Ada Lovelace,
sm_89, WDDM/WSL2 passthrough) unless stated otherwise.

---

## Methodology & Scope

This section states, once, the framing that governs every residue/EI result
in this document. Every experiment section below that measures an Exposure
Index links back here (**"Mandatory framing: see Methodology & Scope"**)
instead of repeating the full text.

**Mandatory framing (verbatim — used in every figure caption and every
residue/EI section in this document):**

> Residue is fully observable when the subsequent kernel is co-resident on
> the same SM — co-residency is forced in this experiment to demonstrate the
> window. This is in-process cross-kernel shared-memory persistence, NOT
> cross-tenant LeftoverLocals on NVIDIA.

**Multi-device extension** (applies wherever more than one GPU is measured —
`multi_gpu/` and the Multi-GPU Replication section):

> Co-residency is forced on every device, so these are controlled-placement
> results consistent with our adversarial threat model — not natural-scheduling
> frequencies, and not a claim about production prevalence or about non-tested
> hardware.

**What "co-residency is forced" means in practice**: single-block launches
with `__launch_bounds__(32,1)` (or an equivalent occupancy cap for
multi-block experiments — see each section's own method) make the listener
kernel's block land on the same SM as the writer's, by construction, not by
chance. This demonstrates that the residue window *exists*; it does not
characterize how often uncontrolled/production scheduling would produce the
same co-residency (see the separate Unforced Co-Residency section for a
measurement of natural, unpinned placement rates).

**What this is not**: none of the measurements in this repository reproduce
cross-tenant LeftoverLocals (CVE-2023-4969) on NVIDIA hardware. NVIDIA was
not reported affected by that specific on-chip-scratchpad, cross-process
attack path. Every experiment here is intra-process (same CUDA context,
consecutive kernel launches) unless a section explicitly states otherwise
(see Unforced Co-Residency's "Not attempted: cross-process / CUDA MPS"
subsection for the one place this boundary was tested and found out of
session scope).

**Hardware caveat**: unless a section states otherwise, all numbers are from
the RTX 4050 Laptop GPU (Ada Lovelace, sm_89, WDDM/WSL2 passthrough). Relative
overhead and qualitative findings (DSE behavior, residue mechanism) generalize;
absolute timing, power, and driver-specific numbers do not transfer to other
hardware (H100, other consumer GPUs) without a fresh measurement. The
Multi-GPU Replication section contains genuine measurements from two
additional devices (RTX 3050 Laptop GPU, RTX 3060 Ti) — those are labeled
per-device and are not blended into the RTX 4050 baseline.

---

## Claim-Mapping Summary

| Paper claim | Paper's original (estimated) value | Measured artifact result | File / Figure |
|---|---|---|---|
| **§6.2 / RP §VI.B** — Recovery rate along the three lifecycle paths (P-reuse, P-spill, P-stage) | ">90% of words" (estimated); RP had `[[MEASURE: P-spill EI]]`, `[[MEASURE: P-stage EI]]` unfilled | **P-reuse: EI 1.000 → 0.000** (N=200, 200/200 same-SM). **P-stage: EI 1.000 → 0.000** (N=200, 200/200 same-alloc). **P-spill: EI 0.000 in ALL phases including unscrubbed** — genuine negative/bounding result, see P-spill subsection below; NOT a "scrub works" claim, there was no exposure to reduce | `results/leakage_demo.json`, `results/leakage_demo_pstage.json`, `results/leakage_demo_pspill.json`; `figures/fig3_exposure_index` |
| **Table I / §6.1** — Scrub overhead for three kernel classes (reduction, GEMM, attention) | "1–5% estimated" (all six cells placeholder) | Host-side CUDA events: **below timing noise floor**, σ upper bounds < 1.08 µs (GEMM), < 1.26 µs (reduction), < 1.77 µs (attention). **Now also resolved with on-device `clock64()` (jitter-free)**: reduction 209.4 ns/exit, GEMM 62.4 ns/exit, attention 164.3 ns/exit — all comfortably inside the CUDA-event upper bounds | `results/benchmark.json`, `results/benchmark_clock64.json`; `figures/fig4_benchmark` |
| **§4.4 / Theorem 1** — DSE hazard: naive zeroing eliminated by compiler; volatile+fence survives | Elimination claimed for naive scrub; survival claimed for volatile+fence | **Ordering differential** (not per-store elimination) on ptxas 12.8 and 13.3 at -O3 sm_89: STS counts equal (58 each); hardened uniquely emits `MEMBAR.SC.CTA` + extra `BAR.SYNC`; pure-dead allocation-granularity DSE confirmed (naive → 0 STS, volatile → 29 STS where buffer never read) | `results/dse_differential.json`; `sass_dumps/dse_test.sass`; `figures/fig2_dse_differential` |
| **§5.4** — Verifier independence: attester re-derives scrub presence from binary without trusting witness fields | "Independent re-derivation" (asserted, not demonstrated) | **T1–T6 tamper-test ladder**: T1–T5 correctly FAIL (hash mismatch, missing scrub in SASS, unaccounted kernel, missing source hash); T6 PASS (known limitation — generate-from-scratch requires binary-level taint to close) | `src/attester.py`; `src/tamper_tests.py`; `figures/fig5_tamper_panel` |
| **RP §VI.E** — Unforced co-residency: natural (unpinned) scheduler placement, not forced single-block pinning | `[[MEASURE: X% of Y trials]]`, `[[MEASURE: unforced EI]]` unfilled | **40.00% natural co-residency rate** (1000 listener blocks/50 reps, no `__launch_bounds__` pinning); of co-resident blocks, **EI 1.000 → 0.000** with LMHC scrub. Rate exactly matches WRITER_NBLOCKS/LISTENER_NBLOCKS=8/20, indicating deterministic (not random) block-to-SM scheduling for this fixed grid geometry. Cross-process/MPS variant NOT attempted — see Unforced Co-Residency section | `results/unforced_coresidency.json` |
| **RP §VI.D** — On-device cycles + amplification bandwidth | `[[MEASURE: cycles/exit per class]]`, `[[MEASURE: ns/exit]]`, `[[MEASURE: ns per word, 95% CI]]` unfilled | `clock64()`-resolved (jitter-free): reduction 209.4 ns/exit, GEMM 62.4 ns/exit, attention 164.3 ns/exit; full-SM amplification: **0.034937 ns/word** | `results/benchmark_clock64.json` |
| **RP §VI.D** — Analytical model, predicted vs. measured | Table V cells unfilled | Calibrated on 2 points (b=126.55 GB/s/SM, L=170.24 cycles); **held-out predictions** for GEMM (1.41×) and attention (0.88×) within a factor of 1.5× of measured | `results/model_validation.json` |
| **RP §VI.F** — Verifier cost (wall-clock, cubin size, CFG-reconstruct time, peak memory) | Table VI cells unfilled | Verify wall-clock ≈0.44–0.51 s, peak RSS ≈96 MB, CFG-reconstruct ≈0.28 s; N=1–8 synthetic-kernel scalability probe shows no visible scaling at this range (fixed-overhead-dominated) | `results/verifier_perf.json` |
| **RP Table III / §VI.A** — Three-GPU (Ampere + Ada) P-reuse replication | Table III listed RTX 3050/3060 Ti/4050 rows with numeric values but ONLY RTX 4050 had ever actually been run (flagged as a contradiction in Phase 0 orientation) | **All three devices now real**: RTX 3050 (Ampere, sm_86): Control 0.000, No-LMHC **1.000**, LMHC 0.000, driver `610.43.02`. RTX 3060 Ti (Ampere, sm_86): Control 0.000, No-LMHC **0.995**, LMHC 0.000. RTX 4050 (Ada, sm_89): Control 0.000, No-LMHC **1.000**, LMHC 0.000. Table III fully backed; driver-version placeholder only fillable for RTX 3050 (other two mislabeled — see Multi-GPU Replication section) | `multi_gpu/results/*.json`, `multi_gpu/results_table.md` |

*Hardware caveat: all measurements on RTX 4050 Laptop GPU (Ada Lovelace, sm_89, WDDM/WSL2 passthrough) unless stated otherwise; the Multi-GPU Replication section below also includes a genuine RTX 3060 Ti (Ampere, sm_86) run. Relative overhead generalises qualitatively; absolute µs values do not transfer to H100 or other targets.*

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

**Mandatory framing**: see [Methodology & Scope](#methodology--scope) above (applies verbatim).

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

### Extended ptxas flag sweep [ADDED — 2026-07-08]

RP §VI.C requires trying more aggressive real ptxas flag combinations before
accepting that per-store elimination doesn't fire on this toolchain. Ran
`dse_test.cu`'s `scrub_naive` (the realistic fill→read→scrub pattern) through:

| Flags | scrub_naive STS count |
|---|---:|
| `-arch=sm_89 -O3` (baseline) | 58 |
| `-O3 -Xptxas -O3,--allow-expensive-optimizations=true` | 58 |
| `-O3 -Xptxas -O3 -maxrregcount=16` | 58 |
| `-O3 -Xptxas -O3 -maxrregcount=8` | 58 |
| `-O3 --use_fast_math` | 58 |
| `-O2` | 58 |
| `-arch=sm_86 -O3` (Ampere target — relevant to RTX 3050/3060 Ti) | 58 |
| `-O3 -dlto` | compile failed — `-dlto` cannot generate a `--cubin` intermediate at compile time (device-LTO requires a link step; not applicable to single-TU `--cubin` builds with this toolchain) |

**No flag combination tried, including targeting sm_86 (Ampere) instead of
sm_89 (Ada), produced per-store elimination of the fill→read→scrub pattern.**
The 58-STS finding is robust across `-O2`/`-O3`, aggressive-optimization and
register-pressure flags, fast-math, and architecture target. This does not
prove no combination anywhere would trigger it — only that the combinations
tried here (a `-maxrregcount`/`--allow-expensive-optimizations`/`--use_fast_math`/
cross-arch sweep) did not.

**Result files**: `src/dse_attempts.sh`, `sass_dumps/dse_flag_sweep/`

### DSE-possibility exhibit: generic C/LLVM analog [ADDED — 2026-07-08]

RP §VI.C separately requires demonstrating the *semantic possibility* of
per-store elimination by lowering the kernel to "a standard optimizer IR" and
running "an aggressive dead-store-elimination pass" — explicitly distinct from
the ptxas-only finding above, and the evidence that makes Theorem 1
non-circular independent of any one back end's current behavior.

**Scope caveat**: nvcc's internal device-side compiler (NVVM, LLVM-based) is
closed-source and not independently invokable with `opt` outside ptxas — we
cannot run `opt -passes=dse` directly on the CUDA kernel's actual internal IR.
Instead, `src/dse_ir_exhibit.c` implements a **generic C analog of the
identical semantic hazard**: a function-local (non-escaping) buffer, filled,
partially read (mirroring `*output = buf[0] ^ buf[NWORDS-1]`), then scrubbed
with either a plain store loop (`fill_read_scrub_naive`) or a
volatile-qualified store loop (`fill_read_scrub_hardened`) — same shape as
`dse_test.cu`'s `scrub_naive`/`scrub_hardened`, same escape-analysis property
(CUDA `__shared__` does not escape the kernel; a C stack-local array does not
escape the function). This is a standard, citable class of hazard (RP ref
[15], Yang et al., USENIX Security 2017) demonstrated with genuine LLVM
tooling (clang/LLVM 21.1.8, installed for this test), not GPU/NVVM-specific.

Three pipelines were tried; only the standard aggressive optimizer pipeline
(`clang -O2`, which runs the full pass pipeline including DSE) produced a
clean result — raw `-O0` IR plus an isolated `opt -passes=dse` (with or
without a `mem2reg`/SROA cleanup pass first) left both functions' stores
untouched, because at that IR level the loop induction variable and
GEP-computed store address are not yet simplified enough for LLVM's DSE to
recognize the redundancy (a known limitation of running DSE in isolation
without the passes that normally precede it in `-O2`).

**`clang -O2 -S -emit-llvm` result** (full IR: `sass_dumps/dse_ir_exhibit.O2.ll`):

| Function | Store instructions in optimized IR | Scrub (`store ... 0`) present? |
|---|---:|---|
| `fill_read_scrub_naive` | 3 (2 vectorized fill-phase stores + 1 volatile store to `sink`) | **No — 0 zero-stores of any kind; the scrub loop is fully eliminated** |
| `fill_read_scrub_hardened` | 11 (2 vectorized fill-phase stores + 1 volatile `sink` store + **8 explicit `store volatile i32 0` instructions**) | **Yes — 8 volatile zero-stores survive intact** |

This is the DSE-possibility exhibit RP §VI.C calls for: a standard, aggressive
optimizer pipeline **does** delete the naive scrub for this exact code shape,
while the identical `volatile`-qualified scrub survives unchanged — confirming
the hazard is real and semantic, not merely hypothetical, even though shipping
ptxas 12.8/13.3 (see sweep above) happens not to exercise it today. This is
reported as corroboration of Theorem 1's permission-level argument, not as a
claim about ptxas's own internals, which remain opaque to this exhibit.

**Result files**: `src/dse_ir_exhibit.c`, `src/dse_ir_build.sh`,
`sass_dumps/dse_ir_exhibit.O0.ll`, `sass_dumps/dse_ir_exhibit.O0.dse.ll`,
`sass_dumps/dse_ir_exhibit.clean.ll`, `sass_dumps/dse_ir_exhibit.clean.dse.ll`,
`sass_dumps/dse_ir_exhibit.O2.ll`

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

### Verifier cost [ADDED — 2026-07-08]

**RP claim** (§VI.F / Table VI): per binary, report wall-clock verification
time, cubin size, per-kernel CFG-reconstruction time, and peak memory; "no
scaling curve is fit to so few points" for the test corpus available.

**Method** (`src/verifier_perf.py`, wraps `attester.py` unmodified —
Module 5's verification logic is not touched): wall-clock via
`time.perf_counter()` around a subprocess run of `attester.py`; peak RSS via
`resource.getrusage(RUSAGE_CHILDREN).ru_maxrss` after that subprocess exits;
CFG-reconstruction time measured in isolation as just the `cuobjdump -sass`
call CHECK 2 depends on; `cubin_size_bytes` surfaced from the compile step.

**A genuine finding surfaced in the process, not a bug**: the on-disk
`results/witness.json` predates the T5 CHECK-1 fix documented in Module 5
above (it has no `source_file_hash` field), so it correctly **FAILs** under
the current (post-fix) `attester.py` — this is exactly the old-format-vs-new-
verifier mismatch RESULTS.md's own T5/T6 analysis describes, encountered
here as a side effect of timing the "existing corpus," not manufactured.
`results/witness.json` was **not** overwritten (RESULTS.md cites its exact
`kernel_hash` value verbatim); instead a fresh, current-format witness was
generated into a temp directory to get the PASS-path cost the RP's Table VI
actually wants.

| Binary | Cubin size | Verify wall-clock | Verify peak RSS | CFG-reconstruct wall-clock |
|---|---:|---:|---:|---:|
| `test_kernels` — **stale on-disk witness (pre-T5-fix format)** | 10,656 B | 0.446 s | 60.1 MB | — |
| `test_kernels` — **fresh witness (current format, PASS)** | 10,656 B | 0.506 s | 96.3 MB | 0.276 s |

### Synthetic scalability test (labeled, not a real-workload claim)

RP §VI.F's instruction to synthesize toy kernels for a scalability probe,
explicitly not fit to a curve at this N: `src/verifier_perf.py` generates
N=1,2,4,8 near-duplicate `gemm_tile`-shaped kernels (each with one
`__confidential__`-tainted buffer) and runs the **real** discover → rewrite →
compile → witness_gen → attester pipeline on each.

| N kernels | Cubin size | witness_gen wall-clock | Verify wall-clock | Verify peak RSS | Verdict |
|---:|---:|---:|---:|---:|---|
| 1 | 4,768 B | 0.581 s | 0.444 s | 96.4 MB | PASS |
| 2 | 8,224 B | 0.454 s | 0.444 s | 96.6 MB | PASS |
| 4 | 15,008 B | 0.717 s | 0.430 s | 96.6 MB | PASS |
| 8 | 28,576 B | 0.718 s | 0.498 s | 96.6 MB | PASS |

**Interpretation**: cubin size scales linearly with kernel count (as
expected — each kernel adds a fixed amount of code). Verify wall-clock does
**not** visibly scale with kernel count over this range (0.43–0.50 s across
an 8× kernel-count increase) — at this small scale, cost is dominated by
fixed per-invocation overhead (Python interpreter startup, one `cuobjdump`
subprocess call) rather than per-kernel verification work. Per Phase 1.6's
own instruction, **no scaling curve is fit** to these four points; whether
verification time grows with kernel count remains an explicit scope limit,
to be characterized on a larger corpus in future work.

**Result files**: `results/verifier_perf.json`, `src/verifier_perf.py`

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

**Mandatory framing**: see [Methodology & Scope](#methodology--scope) above (applies verbatim).

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

### Three-path residue recovery: P-reuse / P-spill / P-stage [ADDED — 2026-07-08]

RP §I distinguishes three physical paths a secret can leak through: **P-reuse**
(on-chip shared memory reused by a later kernel — the only path Module 6
originally measured, above), **P-spill** (register/shared pressure spilled to
CUDA "local" memory, which is per-thread but physically backed by global
DRAM), and **P-stage** (explicit shared→global staging, as tiled GEMM/attention
kernels routinely do). RP §VI.B carried `[[MEASURE: P-spill EI]]` and
`[[MEASURE: P-stage EI]]` as unfilled placeholders. Both paths are now measured,
methodologically parallel to Module 6 (same sentinel `0xCAFEBABE`, same EI
definition, N=200 per phase, plus a control phase not present in the original
Module 6 run).

| Path | Mechanism tested | Control EI | Unscrubbed EI | LMHC-scrubbed EI | Result file |
|---|---|---:|---:|---:|---|
| **P-reuse** | On-chip shared memory reused by next kernel on same SM | 0.000 | **1.000** | **0.000** | `results/leakage_demo.json` |
| **P-stage** | `cudaMalloc`/`cudaFree` reuse of an explicit shared→global staging buffer | 0.000 | **1.000** | **0.000** | `results/leakage_demo_pstage.json` |
| **P-spill** | CUDA "local" (register/shared-spill) memory reuse across kernel launches | 0.000 | **0.000** | 0.000 | `results/leakage_demo_pspill.json` |

**P-stage (positive result, same shape as P-reuse).** A writer kernel stages a
16 KB shared tile into a freshly `cudaMalloc`'d global buffer, frees it
unscrubbed; a scavenger's next `cudaMalloc` of the same size received the
identical pointer in 200/200 trials (`same_alloc_trials: 200`, `diff_alloc_trials: 0`)
and recovered the full pattern (EI 1.000). With the LMHC hardened scrub (volatile
global-store loop + `__threadfence()`) applied to the staging buffer before free,
EI → 0.000. This directly supports RP §VI.B's P-stage claim on the RTX 4050.
Caveat: the 200/200 allocator-reuse rate reflects a single-threaded process with
no intervening allocations of a different size — a realistic adversary would
need the same allocation-size/timing conditions; this is not a claim that
`cudaMalloc` always returns a just-freed block.

**P-spill (negative/bounding result — reported honestly, not suppressed).**
The same writer/listener design applied to a genuinely-spilled per-thread local
array (256 × uint32/thread, confirmed via `cuobjdump -sass` — see
`sass_dumps/pspill_test.sass` — to lower to real `STL`/`LDL` instructions in a
256-iteration loop, not registers, not dead-code-eliminated) shows **EI = 0.000
in every phase, including the unscrubbed writer and a same-kernel-twice
diagnostic** (`src/pspill_diag.cu`). A follow-up diagnostic
(`src/pspill_diag2.cu`, output in `results/pspill_diagnostics.txt`) confirms the
listener reads the *identical virtual address* the writer wrote (`same_address=YES`)
but observes zero, not the pattern (`residue_present=NO`), across three repeated
runs. **This bounds RP Platform Assumption A1 (§III.A) for the P-spill path
specifically**: A1 states scratchpad SRAM *and* freed spill/global backing are
"deallocated but not zeroed" — that holds for on-chip shared memory (P-reuse,
100% residue under the identical software stack) but was **not observed to hold
for CUDA local memory** on this hardware/driver/toolchain (RTX 4050, driver
573.22, CUDA 12.8 runtime, **WSL2/WDDM GPU passthrough**). Root cause is not
established by this measurement — candidates include WDDM/WSL2-layer page
zeroing on local-memory (re)allocation, versus a genuine driver/architecture
behavior that would also hold on native Linux. **This needs a native-Linux
(non-WSL2) run to disambiguate before the RP can state anything about P-spill
prevalence beyond "not observed on this configuration."** Do not read this as
"P-spill is safe on NVIDIA" — it is a bound on one measurement, not a proof of
absence, and directly contradicts nothing else in this artifact since P-reuse
and P-stage both show full residue under the same software stack.

**Result files**: `results/leakage_demo_pstage.json`, `results/leakage_demo_pspill.json`, `results/pspill_diagnostics.txt`, `sass_dumps/pspill_test.sass`

---

## Unforced Co-Residency [ADDED — 2026-07-08]

**Paper claim** (RP §VI.E): "Dropping the launch pinning, we ran a victim and
a separate scavenger workload... as multi-block kernels under the default
scheduler... and measured how often the scavenger naturally co-resides with
the victim and recovers residue: `[[MEASURE: X% of Y trials]]` recovered
non-zero residue (mean EI `[[MEASURE: unforced EI]]`). With the scrub enabled
the identical experiment yields zero recovery (`[[MEASURE: 0/Y trials]]`, EI 0)."

**SCOPE GUARD (mandatory, per [`INTERPRETATION.md`](INTERPRETATION.md) and
`multi_gpu/README.md`'s own pre-existing commitment):** those files state this
artifact performs "controlled-placement replication only" and that an
unconstrained/natural-scheduling mode is "explicitly out of scope," while also
providing the escape hatch: *"If such a mode is ever added later, it must be a
separate exploratory path that never alters or replaces the controlled
results, and its outputs must be labeled device-specific and
non-generalizing."* `src/unforced_coresidency.cu` is that separate path. It
does **not** modify `leakage_demo.cu`, `multi_gpu_repro.cu`, or any of their
result files. Its output lives only in `results/unforced_coresidency.json`
and this section.

**Design**: same OS process and CUDA context as the rest of this artifact —
this is NOT a cross-process/cross-context test (see Not Attempted, below). The
difference from Phase 0/Module 6 is that **placement is not forced**: no
`__launch_bounds__` pinning, and a multi-block grid instead of a single block.
A writer kernel launches 8 blocks (`WRITER_NBLOCKS`); a listener kernel
launches 20 blocks (`LISTENER_NBLOCKS`, matching the RTX 4050's 20 SMs). Each
block uses 62.5 KB of *dynamic* shared memory (`cudaFuncSetAttribute(...,
cudaFuncAttributeMaxDynamicSharedMemorySize, ...)`, since static `__shared__`
arrays cap at 48 KB/block on sm_89) — large enough that at most one block is
resident per SM at a time, so occupancy doesn't trivially force every SM to be
touched by both kernels (an earlier draft with 64 small blocks made
co-residency tautologically 100% and was discarded — see git history of this
file for the corrected design). For every listener block, `%smid` is compared
against the set of SMIDs the writer's blocks used in the same repeat; 50
repeats, 1000 listener-block observations total.

### Measured result

| Phase | Natural co-residency rate | EI of co-resident blocks |
|---|---:|---:|
| A — writer WITHOUT LMHC scrub | 400/1000 = **40.00%** | **1.000000** |
| B — writer WITH LMHC hardened scrub | 400/1000 = **40.00%** | **0.000000** |

**Fills the RP's placeholders**: `[[MEASURE: X% of Y trials]]` → 400/1000
(40.00%); `[[MEASURE: unforced EI]]` → 1.000000; `[[MEASURE: 0/Y trials]]` →
0/1000 nonzero-word co-resident blocks in Phase B (EI 0.000000).

**Interpretation**: The co-residency rate (40.00%) is **exactly**
`WRITER_NBLOCKS / LISTENER_NBLOCKS = 8/20`, reproduced identically across 50
repeats in both phases. This indicates the CUDA scheduler's block-to-SM
assignment is **deterministic for a fixed grid geometry** on this hardware —
not a random/entropy-bearing process — at least for these grid sizes and
occupancy settings. When a listener block does land on an SM the writer used,
recovery is total (EI 1.0), exactly matching the forced-co-residency result
(Phase 0 / Module 6) — natural, unpinned scheduling does not attenuate the
residue in any way once co-residency occurs; the LMHC scrub closes it just as
completely (EI → 0.0).

**Caveat (mandatory, do not drop):** `natural_coresidency_rate` here is an
artifact of this exact grid configuration (8 writer blocks, 20 listener
blocks, 62.5 KB/block shared footprint, 20 SMs) on this specific device and
driver. It is **not** a general "40% of the time GPUs leak residue" claim —
different grid sizes, occupancy, or hardware would almost certainly produce a
different deterministic ratio. Report only as measured, exactly as
[`INTERPRETATION.md`](INTERPRETATION.md)'s existing outcome-reading discipline requires
for the forced-co-residency table.

### Not attempted: cross-process / CUDA MPS

RP §VI.E also describes "two independently launched processes of the same
user" and, separately, "two clients of one MPS context." **Neither was
attempted in this session.** What's attempted above stays within one process
and one CUDA context (placement is unforced, but the context boundary is not
crossed). Two-process residue under NVIDIA's *default* (non-MPS) driver
behavior is expected, per the LeftoverLocals paper's own findings and NVIDIA's
per-context resource isolation, to show little or no residue — but that is an
expectation, not a measurement, and asserting it without running it would
violate this artifact's own no-fabrication rule. A genuine MPS-based
measurement requires starting `nvidia-cuda-mps-control` and confirming it
functions correctly under WSL2/WDDM GPU passthrough (uncertain — MPS support
in virtualized/paravirtualized GPU passthrough environments is not guaranteed
by NVIDIA and was not verified before this session ended). **This is flagged
as a genuine gap for either a follow-up session with time to validate MPS
under WSL2, or for the RP text to descope the cross-process/MPS sentence
specifically** (distinct from the same-process unforced result above, which
IS measured and can support the rest of §VI.E as written).

**Result file**: `results/unforced_coresidency.json`

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

### On-device clock64() resolution [ADDED — 2026-07-08]

RP §VI.D item (a) asks for "on-device cycles: bracketing the epilogue with
`clock64()`/`%globaltimer`" to get `[[MEASURE: cycles/exit per class]]` and
`[[MEASURE: ns/exit]]`, jitter-free — precisely because the CUDA-event method
above could only bound, not resolve, the scrub cost. `src/benchmark_clock64.cu`
brackets just the scrub region (`__syncthreads() -> t0=clock64() -> [scrub
loop + threadfence_block + syncthreads] -> t1=clock64()`) for the same three
kernel geometries as `benchmark.cu` (N=500 reps, 50 warmup). Unlike host-side
timing, `clock64()` cycle counts came back essentially noise-free (stddev ≈ 0
for reduction; ≤0.13 cycles for attention/gemm) — this is a genuinely resolved
measurement, not another bound.

| Kernel class | Scrub words | Cycles (median) | ns/exit (median) |
|---|---:|---:|---:|
| Reduction | 4096 | 446.0 | **209.39 ns** |
| GEMM | 256 | 133.0 | **62.44 ns** |
| Attention | 2048 | 350.0 | **164.32 ns** |

All three are, as expected, comfortably inside the CUDA-event method's σ
upper bounds (1.08–1.77 µs) — confirming the earlier upper-bound framing was
correct, and now supplying the actual point estimate the RP asks for. ns/exit
does not scale linearly with word count alone (GEMM: 256 words/256 threads =
1 word/thread vs. reduction: 4096 words/256 threads = 16 words/thread vs.
attention: 2048 words/64 threads = 32 words/thread) — thread count and
stores-per-thread both matter, which is why the amplification method below
reports a single canonical bandwidth figure instead.

**Caveat**: `clock64()` counts SM clock cycles exactly; the ns conversion
uses `cudaDeviceProp.clockRate` queried once at idle before the runs, so the
ns figures (not the cycle counts) inherit whatever imprecision GPU
Boost/dynamic clocking introduces relative to that fixed reference.

### Full-SM-footprint amplification [ADDED — 2026-07-08]

RP §VI.D item (b): scrubbing a footprint near the ~100 KB/SM ceiling
(24,000 words = 93.75 KB, 1024-thread block — the max threads/block on
sm_89) and dividing by word count gives a bandwidth-representative ns/word
figure independent of any one kernel's thread geometry, filling
`[[MEASURE: ns per word, 95% CI]]`.

**Result: 1786.0 cycles = 838.50 ns total = 0.034937 ns/word (95% CI ≈ ±0.000000
— clock64() gave a stddev of 0.0 cycles across 500 reps for this measurement,
so the confidence interval on the mean is not distinguishable from zero at
this precision; this reflects the counter's determinism for a single-block,
warp-synchronous kernel, not an implausibly perfect measurement).**

**CUPTI/Nsight Compute**: not attempted — `ncu`/Nsight Compute is not part of
the `cuda-nvcc`/`cuda-cuobjdump` apt packages installed for this artifact and
was not separately installed. Not pursued because `clock64()` bracketing
already resolved the measurement to the precision RP §VI.D needs; a hardware
profiler would add corroboration but is not required to fill any remaining
placeholder.

**Result files**: `results/benchmark_clock64.json`, `src/benchmark_clock64.cu`

### Energy / power draw [ADDED — 2026-07-08]

The original build (SPEC.md, RESULTS.md pre-2026-07-08) stated: *"We did not
instrument power telemetry ... so we make no energy claim."* `nvidia-smi
--query-gpu=power.draw` is in fact available in this WSL2/driver stack, so a
genuine (if coarse) measurement was taken rather than leaving the gap
unaddressed.

**Method** (`src/power_sustained.cu`, `src/power_measure.sh`): `nvidia-smi
--query-gpu=power.draw --format=csv,noheader -lms 100` sampled in parallel
with a ~4-second sustained CUDA launch loop for three modes — idle (no GPU
work), empty (fill-only kernel, no scrub, ~472,600 launches/4s), and scrub
(identical fill plus the full LMHC hardened scrub of a ~93.75 KB footprint,
~466,800 launches/4s — same geometry as the amplification kernel in
`benchmark_clock64.cu`).

| Mode | Mean power | σ |
|---|---:|---:|
| Idle | 14.24 W | 0.12 |
| Empty (fill only, sustained) | 17.37 W | 1.76 |
| Scrub (fill + LMHC scrub, sustained) | 21.87 W | 2.75 |

**Delta: scrub adds 4.50 W over empty-only under sustained back-to-back
launching (~26% relative increase over the empty-kernel sustained draw).**

**Mandatory caveat — do not overread this number**: this isolates the power
cost of running scrub-heavy kernels **continuously back-to-back at ~117,000
launches/sec**, not the per-exit energy of a single scrub inside an
otherwise-idle production pipeline. `nvidia-smi` reports whole-board power at
~10 Hz — four to five orders of magnitude coarser than the scrub's actual
duration (200–800 ns, per `benchmark_clock64.json`). This is the coarsest
instrument available without CUPTI power counters (not installed, same gap
noted for Nsight Compute above). **The one thing worth flagging plainly**:
the relative power delta (~26%) is proportionally much larger than the
relative timing overhead found via `clock64()` (all three kernel classes
were a small fraction of total kernel time) — under sustained saturation,
scrub-related power draw is not obviously "free" just because scrub latency
is small, and this deserves a real per-exit energy measurement (CUPTI power
counters) before any energy claim is made in the paper text.

**Result files**: `results/power_sustained.json`, `src/power_sustained.cu`, `src/power_measure.sh`

---

## Analytical Cost Model Validation [ADDED — 2026-07-08]

**RP claim** (§VI.D, "An analytical model"): `C_scrub ≈ F / min(b, P·w·f) + L`
predicts per-exit scrub cost from footprint bytes (F), address-space
bandwidth (b), resident threads (P), store width (w), issue rate (f), and
barrier cost (L); Table V asks whether the prediction falls within the
measured bound for reduction, GEMM, and attention.

**Method** (`src/model_predict.py`, real measured input, no invented
numbers): the model reduces to `cycles = F/b + L` in the bandwidth-bound
regime the RP itself says should dominate for large F. Rather than curve-fit
all four measured points (which would prove nothing), **b and L are solved
exactly from two calibration points** — `full_sm_amplification` and
`reduction`, the two largest-F measurements from `results/benchmark_clock64.json`
— and then used to **predict** GEMM and attention as **held-out** data,
comparing prediction to the independent `clock64()` measurement.

### Calibration

| Parameter | Fitted value |
|---|---|
| b (shared-store bandwidth) | 59.41 bytes/cycle = **126.55 GB/s** per SM |
| L (barrier/epilogue cost) | 170.24 cycles = 79.9 ns |

The fitted bandwidth (126.55 GB/s/SM) lands close to a commonly cited
back-of-envelope figure for Ada-class shared-memory bandwidth (~128
bytes/clock/SM, i.e. one 4-byte word per lane per cycle with no bank
conflicts, at this device's 2.13 GHz clock) — reported as corroboration of
the fit's plausibility, not as a target the fit was adjusted to match (the
figure was computed once, from the two calibration points, and not tuned).

### Held-out prediction vs. measurement

| Kernel | Role | Predicted (ns) | Measured (ns) | Predicted/Measured |
|---|---|---:|---:|---:|
| Full-SM amplification | calibration | 838.50 | 838.50 | 1.000 (exact, by construction) |
| Reduction | calibration | 209.39 | 209.39 | 1.000 (exact, by construction) |
| **GEMM** | **held-out prediction** | **88.02** | **62.44** | **1.41×** |
| **Attention** | **held-out prediction** | **144.66** | **164.32** | **0.88×** |

**Fills RP Table V**: both held-out predictions land within a factor of 1.5×
of the measured value (GEMM over-predicts by 41%, attention under-predicts by
12%) — the simple two-parameter bandwidth+barrier model is directionally
correct but not exact for these smaller-thread-count kernels (GEMM: 256
threads/256 words = 1 word/thread; attention: 64 threads/2048 words = 32
words/thread), which the RP's own model anticipates: *"thread-bound only when
the block is small."* Reduction (256 threads/4096 words) and the
full-SM-amplification kernel (1024 threads) both sit in the large-F,
bandwidth-bound regime the calibration assumes, by construction. This is
reported as an honest **directional validation** (model predicts the right
order of magnitude and the right regime distinction) rather than a
false-precision claim of exact predictive accuracy — consistent with RP
§VI.D's own instruction to report "within the measured upper bound" rather
than a percent-error against an unresolved number; here, uniquely, we have
resolved point estimates (via `clock64()`) rather than upper bounds, so the
comparison is a direct predicted-vs-measured ratio instead.

**Result files**: `results/model_validation.json`, `src/model_predict.py`

---

## Multi-GPU Replication [UPDATED — 2026-07-08]

**History this session**: at Phase 0 orientation, only RTX 4050 had ever been
run through `multi_gpu/`; a file claimed to be "RTX 3050 Ti" had three
mutually contradictory identities (filename, JSON `hardware` field, and
terminal-banner screenshot all disagreed) and used the wrong JSON schema —
flagged rather than integrated, collaborator asked to re-run via
`multi_gpu/run.sh`. A correctly-schema'd RTX 3060 Ti file arrived mid-session
and was integrated. **A correctly-schema'd RTX 3050 Laptop GPU file has now
also arrived, completing all three devices RP Table III names.**

### Aggregated result (`python3 multi_gpu/aggregate.py`, real output, not hand-typed)

| GPU | Architecture | Driver (as recorded) | CUDA Runtime | Same-SM trials | EI (no scrub) | Control EI | EI (LMHC scrub) | Valid |
|---|---|---|---|---:|---:|---:|---:|---|
| NVIDIA GeForce RTX 3050 Laptop GPU | Ampere (sm_86) | **`610.43.02`** | 12.4 | 200 | **1.0000** | 0.0000 | 0.0000 | Y |
| NVIDIA GeForce RTX 3060 Ti | Ampere (sm_86) | `cuda_api_13.2` (mislabeled, see below) | 12.8 | 200 | **0.9950** | 0.0000 | 0.0000 | Y |
| NVIDIA GeForce RTX 4050 Laptop GPU | Ada Lovelace (sm_89) | `cuda_api_12.8` (mislabeled, see below) | 12.8 | 200 | **1.0000** | 0.0000 | 0.0000 | Y |

**This fills RP Table III completely**, and — for the RTX 3050 row only —
genuinely fills the `[[MEASURE: driver versions]]` placeholder (`610.43.02`,
a real NVIDIA driver string, unlike the other two rows; see below).

Per [`INTERPRETATION.md`](INTERPRETATION.md)'s pre-committed reading (written before any
collaborator data existed): this is **Outcome A** — all three tested devices
show residue without scrub and EI = 0.0 after LMHC. Each device's result
stands on its own; per that document's own instruction, do **not** use
language like "all NVIDIA GPUs" or "modern GPUs" — refer only to these three
device names. Two architectures (Ampere: RTX 3050, RTX 3060 Ti; Ada: RTX
4050) are now genuinely represented, matching RP §VI.A's "two architectures"
claim.

**RTX 3060 Ti shows EI = 0.995, not 1.000 — reported exactly as measured, not
rounded up or explained away.** 4,096 of 819,200 words read zero even without
the LMHC scrub (control run is clean at 0.000, ruling out self-contamination
as the cause). This is a genuine, mild deviation from RTX 4050's and RTX
3050's clean 1.000. Candidate explanations (not established by this single
run): occasional non-co-resident trials slipping past the `%smid` gate at a
boundary case, or ordinary measurement noise at N=200 — RTX 3050 is also
Ampere/sm_86 and shows a clean 1.000, so this is not simply an
architecture-wide Ampere effect. Per `INTERPRETATION.md`'s Outcome-B/C
discipline, this is not treated as a failure or suppressed — it is reported
as measured, and would benefit from a higher-N re-run before being read as
more than a data point.

### `driver_version` field is mislabeled for two of three devices (pre-existing harness bug)

The RTX 3060 Ti and RTX 4050 JSON files record `driver_version` as
`"cuda_api_13.2"` / `"cuda_api_12.8"` — this is the **CUDA Driver API
version** (`cudaDriverGetVersion`), not an actual NVIDIA driver string.
**The RTX 3050 file is different: its `driver_version` field correctly holds
`"610.43.02"`, a genuine driver version string** — so the bug is
non-deterministic or environment-dependent (possibly whether
`/proc/driver/nvidia/version` is readable/present in that particular
WSL2/driver install; `multi_gpu_repro.cu` reads that file for this field, and
it evidently succeeded on the RTX 3050 run's machine but silently fell back
to the CUDA API version on the other two). **RP §VI.A's `[[MEASURE: driver
versions]]` placeholder can be filled honestly only for RTX 3050
(`610.43.02`)**; the other two need either a harness fix (make the
`/proc/driver/nvidia/version` parse failure loud instead of silently
substituting the CUDA API version) and a re-run, or the real driver version
supplied by whoever ran those two collection sessions. Flagged here rather
than silently relabeling or guessing, since `multi_gpu_repro.cu` is shared
collaborator-facing infrastructure and this is now a confirmed intermittent
bug, not a constant mislabeling.

### Not done

- No comparison figure regenerated for the three-device table (Phase 1.1
  asked for one restyled in the "Phase 3.4(e) palette" — that phase/palette
  specification was not included in what was provided to this session; a
  chart can be added once that spec is available, or using `fig3`'s existing
  Okabe-Ito palette as a reasonable default if that's acceptable).

**Result files**: `multi_gpu/results/*.json` (3 devices), `multi_gpu/results_table.md`, `multi_gpu/results_table.tsv`, `multi_gpu/results_table.html`
