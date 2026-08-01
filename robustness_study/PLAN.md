# PLAN.md — UPES Robustness Study, Phase 1 (Study Design)

Status: draft design, finalized pending the two feasibility compiles in §9
and your review. Builds on `AUDIT.md` (Phase 0). Answers already confirmed:
sm_87 substitutes for sm_70; WSL2 is the sole execution environment.

Research question this supports (from the task brief): *when zeroization is
expressed in source, under what compiler/optimization/context/toolkit/
architecture conditions does the required structure survive into the shipped
binary, and how reliably can that structure be recognized from the binary
alone?*

---

## 1. Zeroization constructs (Z0–Z6)

All constructs operate on a single designated `__shared__` buffer, referred
to below as `SBUF` (word count `NWORDS`, `uint32_t` elements), using a
per-thread strided loop `for (i = tid; i < NWORDS; i += BLOCK) ...` where
`BLOCK == blockDim.x` for that kernel. **`NWORDS` is always chosen so that
`NWORDS % BLOCK == 0`** for every context it's paired with — this guarantees
every thread executes the same number of loop iterations, so no construct
ever places a barrier on a path some threads can skip. This is verified
per-context in §2, not assumed.

| ID | Name | Code shape appended after the context's "live" phase | Rationale | Expected structural requirement | Role |
|----|------|------|------|------|------|
| **Z0** | No scrub | *(nothing — SBUF is simply left as-is)* | Negative control: establishes the ABSENT baseline the classifier must correctly recognize | none | Negative control |
| **Z1** | Plain zeroing loop | `for (...) SBUF[i] = 0u;` (plain pointer) | The literal naive form the original repo already studies at K2/sm_89/O3 only; this study asks whether that "DSE doesn't fire" finding holds across contexts/archs/toolkits/opt-levels it was never tested on | zero-stores present or eliminated, depending on compiler decision — this is exactly what we're measuring | Primary variable under test |
| **Z2** | Volatile zeroing loop | `volatile uint32_t *v = SBUF; for (...) v[i] = 0u;` — no fence, no extra barrier beyond whatever the context already required | Isolates whether `volatile` alone (no ordering guarantee) changes survivability vs. Z1 | zero-stores present (volatile forbids elimination under the C++ abstract machine); no fence/ordering signature expected | Partial case — store presence without ordering guarantee |
| **Z3** | Plain loop + final `__syncthreads()` | `for (...) SBUF[i] = 0u;` then `__syncthreads();` | Isolates whether a trailing barrier alone (no volatile) changes what ptxas keeps, independent of the volatile question | zero-stores present or eliminated (same DSE question as Z1) plus a `BAR.SYNC`; no `MEMBAR.SC.CTA` expected | Partial case — tests barrier-only effect |
| **Z4** | Volatile loop + final `__syncthreads()` | `volatile uint32_t *v = SBUF; for (...) v[i]=0u;` then `__syncthreads();` | Isolates store-survival-via-volatile plus cooperative completion, without the explicit fence | zero-stores present; `BAR.SYNC` present; no `MEMBAR.SC.CTA` | Partial case — store + barrier, no fence |
| **Z5** | Hardened (LMHC-style) | `volatile uint32_t *v = SBUF; for(...) v[i]=0u; __threadfence_block(); __syncthreads();` | The construct the original repo's `scrub_hardened` already uses — carried forward as the primary "should satisfy full policy" construct, now tested across the full new axis set | zero-stores + `MEMBAR.SC.CTA` + `BAR.SYNC`, in that order, before `EXIT` | Intended FULL_SIGNATURE |
| **Z6** | Inline-PTX hardened | per-word `unsigned a = __cvta_generic_to_shared(SBUF+i); asm volatile("st.shared.u32 [%0], %1;" :: "r"(a),"r"(0u) : "memory");` loop, then `__threadfence_block(); __syncthreads();` | The original repo's `scrub_ptx_fallback` — direct `st.shared` that ptxas cannot legally eliminate regardless of allocation-level analysis; carried forward as a second "should satisfy full policy" construct to check whether it stays indistinguishable from Z5 outside sm_89/12.8-O3 | zero-stores + `MEMBAR.SC.CTA` + `BAR.SYNC`, same as Z5 | Intended FULL_SIGNATURE |

**CUDA-validity notes** (checked while designing, re-verified by compile in
§9):
- All constructs use a fixed trip-count loop (`NWORDS/BLOCK` iterations per
  thread, identical across threads) — no divergent loop bounds.
- `__threadfence_block()` and `__syncthreads()` in Z3–Z6 are called
  unconditionally by every thread in the block, never inside a
  thread-ID-dependent `if`.
- Z0's "do nothing" is still a fully valid kernel — SBUF is simply an
  otherwise-live-or-dead buffer depending on context; nothing appended.

`memset`-style device intrinsics were considered and **excluded**: CUDA has
no single-instruction device-side `__shared__` memset primitive equivalent
in guarantees to a manual loop, and introducing one would compare a
different code-generation path rather than a zeroization-intent variant,
which the task explicitly warns against unless fairness is confirmed. Not
pursued for the core corpus.

---

## 2. Kernel contexts (K0–K5)

Each context defines: a launch geometry (`BLOCK` threads, single block),
zero or more "live" shared buffers used for realistic computation, and the
designated `SBUF`/`NWORDS` that the Z-construct is appended to, always
**after** the last point SBUF is used for anything else in the kernel and
**before** the kernel's single exit. Per-context `NWORDS`/`BLOCK` and the
divisibility check:

| ID | Name | BLOCK | SBUF role / NWORDS | NWORDS % BLOCK | Based on |
|----|------|------:|---|:---:|---|
| **K0** | Pure dead allocation | 32 | SBUF declared, touched only by the Z-construct (if any); never read anywhere | 1024 % 32 = 0 ✓ | `pure_dead_naive`/`pure_dead_volatile` (`dse_minimal.cu`) |
| **K1** | Fill then scrub | 32 | SBUF filled with a tid-derived pattern in Phase 1; kernel's single output is computed **without** reading SBUF (`if(tid==0) *output = tid;`); Z-construct appended after fill | 1024 % 32 = 0 ✓ | Fill phase of `nosync_naive`, output-independence per task's explicit K1 definition (differs from `nosync_*`, which does read SBUF for output) |
| **K2** | Fill, use, then scrub | 32 | SBUF filled (Phase 1), `__syncthreads()`, read to produce `*output` (Phase 2), `__syncthreads()`, Z-construct (Phase 3) | 1024 % 32 = 0 ✓ | `dse_test.cu`'s `scrub_naive`/`scrub_hardened` exactly |
| **K3** | Reduction-like | 256 | SBUF **is** the reduction buffer (`smem_partial`): filled from `input`, tree-reduced with `__syncthreads()` each stride halving, `output[blockIdx.x]=SBUF[0]` at `tid==0`, then Z-construct on SBUF before the single exit | 256 % 256 = 0 ✓ (1 iter/thread) | `reduction_sum` (`test_kernels.cu`) |
| **K4** | Tiled-GEMM-like | 256 (16×16, flattened `tid = ty*16+tx`) | Two tile buffers per the original (`a_tile` confidential-sourced, `b_tile` not); Z-construct applied **only to `a_tile`** (the sensitive one, matching the original's own taint distinction) immediately after the accumulation loop ends and before the divergent `if(row<M&&col<N)` output write | 256 % 256 = 0 ✓ | `gemm_tile` (`test_kernels.cu`) |
| **K5** | Attention-style | 64 (`= ATTN_SEQ`) | `smem_qk` (2048 words, the confidential-sourced buffer) filled from `q_proj`, used to compute `smem_softmax`/`attn_out`, `__syncthreads()`, then Z-construct on `smem_qk` before the single exit | 2048 % 64 = 0 ✓ | `attention_fwd` (`test_kernels.cu`) |

All six are **single-exit, single-block**, and every `__syncthreads()` /
`__threadfence_block()` call site (both the context's own and the
Z-construct's) is reached unconditionally by every thread in the block —
confirmed by inspection of each template above; K4's only divergent
conditional (`if(row<M&&col<N)`) is strictly after the last barrier, and
K5's per-thread conditionals (`if(tid<ATTN_HEAD)`, `if(tid==0)`) wrap only
non-barrier statements in the original source, preserved unchanged.

Multi-exit / early-return kernels are **not** part of the core corpus (task
§8: "must be generated separately as controlled negative or unsupported
cases," §25: never executed on hardware). They appear only in the binary-
detection negative corpus as N10 (compile/disassemble only, never launched).

---

## 3. Core compilation matrix

Axes:
- **Construct**: Z0, Z1, Z2, Z3, Z4, Z5, Z6 (7)
- **Context**: K0, K1, K2, K3, K4, K5 (6)
- **ptxas optimization**: `-Xptxas -O0/-O1/-O2/-O3`, host `nvcc -O3` fixed
  and recorded separately per the task's mandatory host/device split (4)
- **Architecture target**: `sm_75, sm_80, sm_86, sm_87, sm_89, sm_90` — the
  verified intersection of both toolkits' `--list-gpu-arch` output (6)
- **Toolkit**: CUDA 12.8 (nvcc/ptxas/cuobjdump V12.8.9x) and CUDA 13.3
  (V13.3.3x), both in WSL2 (2)

**7 × 6 × 4 × 6 × 2 = 2,016 artifacts.**

Only `sm_89` has matching physical hardware (RTX 4050 Laptop). All other
five targets are **compile-target evidence only** — every record for those
targets is tagged accordingly, and no runtime claim is ever made for them.

**Excluded from the core matrix** (not "silently removed" — explicitly
excluded by design, distinct from a compile failure): `--maxrregcount`,
`--use_fast_math`, `--allow-expensive-optimizations`. The original repo's
own flag sweep (`docs/RESULTS.md`, "Extended ptxas flag sweep") already
tested all three against the K2/sm_89/ptxas-12.8-O3 cell and found zero
effect on STS count. Per the task's own instruction not to inflate the
matrix with flags of no plausible relevance, these are **not** part of the
2,016-cell core design. They will instead get a small, explicit pilot check
(§8) across a couple of new construct/context combinations the original
sweep never touched (Z1×K0 and Z2×K3), specifically to test whether that
"no effect" finding generalizes — if the pilot check shows a real effect
anywhere, I will report that and revisit this exclusion before the core run,
not silently keep it excluded.

---

## 4. Negative / binary-detection corpus (N0–N12)

Separate from the core corpus — evaluates the binary structural-signature
checker (§6), not compiler survivability. Built by direct SASS/source
construction (not the Z/K generator), one artifact each, sm_89/CUDA 12.8
unless noted, using K2 as the host context (realistic fill→use→scrub) except
where the case requires otherwise:

| ID | Case | How constructed |
|---|---|---|
| N0 | No scrub | = Z0×K2 core-matrix cell (reused, not duplicated) |
| N1 | Complete hardened scrub | = Z5×K2 core-matrix cell (reused) |
| N2 | Zero stores present, fence missing | = Z4×K2 (volatile+barrier, no fence) — reused from core matrix |
| N3 | Zero stores + fence present, final barrier missing | Z5 variant with the trailing `__syncthreads()` removed (fence still present) — new, checker-corpus-only source |
| N4 | Barrier present, scrub stores absent | Z0×K2 with a spurious trailing `__threadfence_block(); __syncthreads();` and no zero-store loop — new source |
| N5 | Scrub instructions present but wrong order | Source emits `__syncthreads()` **before** the volatile zero-store loop, then `__threadfence_block()` after — new source, deliberately mis-ordered |
| N6 | One relevant exit uncovered | A controlled two-kernel cubin where only one of two kernels in `sensitive_regions` gets the Z5 treatment, the other gets Z0 — tests whether the checker correctly flags the unscrubbed one rather than passing the cubin as a whole |
| N7 | Nonzero store replacing zero store | Z5 variant storing a nonzero sentinel (`0xAAu`) instead of `0u` — new source; tests whether the checker's `STS…RZ` (zero-register) pattern correctly distinguishes a zero-store from an arbitrary store |
| N8 | Full signature plus unrelated stores | Z5×K2 with an extra unrelated shared buffer written (not zeroed, not the nominated sensitive one) — tests false-positive resistance on noisy binaries |
| N9 | Unsupported/unusual instruction encoding | The same Z5 construct compiled for `sm_90` or `sm_75`, where the SASS mnemonic for a zero-register store may differ textually — feeds directly into the "architecture-aware pattern" requirement (§11 of the task) |
| N10 | Multi-exit control flow | A deliberately early-returning kernel (`if (cond) return;` before the scrub) — **compiled and disassembled only, never launched**, per the task's explicit prohibition on running barrier-unsafe multi-exit kernels on hardware |
| N11 | Kernel missing from binary | `sensitive_regions` names a kernel not present in the compiled cubin (typo / stale witness scenario) |
| N12 | Compilation removed pure-dead plain scrub | = Z1×K0 core-matrix cell (reused) — this is the one cell where the original repo's own evidence (`pure_dead_naive` → 0 STS, `#550-D` warning) predicts elimination, giving a genuine ABSENT ground-truth case for free |

N0–N2, N12 reuse core-matrix artifacts by reference (no duplicate
compilation); N3–N9, N11 require small dedicated source files; N10 requires
one dedicated multi-exit source file, compiled but never executed.

---

## 5. Classification categories

Exactly the task's six categories, no additions: `FULL_SIGNATURE`,
`STORE_ONLY`, `PARTIAL_SIGNATURE`, `ABSENT`, `UNDECIDABLE`,
`COMPILE_FAILED`, `COMPILE_UNSUPPORTED` (seven, counting the two compile-
level outcomes). Decision logic (applied by `classify_binary.py` in Phase 2,
operating only on extracted SASS features — never on `source_intent_label`):

1. Kernel not found in cubin at all → `UNDECIDABLE` (not `ABSENT` — absence
   of the kernel itself is a different failure mode than absence of
   zeroization evidence within a present kernel; task §12 forbids conflating
   "no recognized evidence" with "proof of vulnerability," and a missing
   kernel is not evidence of anything).
2. No zero-valued shared-store evidence found → `ABSENT`.
3. Zero-store evidence found, no `MEMBAR.SC.CTA` and no `BAR.SYNC` after it
   → `STORE_ONLY`.
4. Zero-store evidence found plus **some but not all** of
   {`MEMBAR.SC.CTA`, `BAR.SYNC`} in the required relative order → `PARTIAL_SIGNATURE`.
5. Zero-store evidence, `MEMBAR.SC.CTA`, and `BAR.SYNC` all present in the
   required order, before the (single, checked) `EXIT` → `FULL_SIGNATURE`.
6. Parser encounters an instruction pattern it cannot classify confidently
   (e.g., N9-style unfamiliar encoding, or a multi-exit kernel where exit
   coverage can't be established under this study's checker scope) →
   `UNDECIDABLE`, never silently coerced to `PASS`/`FULL_SIGNATURE` or to
   `ABSENT`.
7. Tool accepted the config but the compile step itself failed →
   `COMPILE_FAILED`, with exact stderr preserved. Tool/target combination
   rejected outright (e.g., an arch string the toolkit doesn't recognize) →
   `COMPILE_UNSUPPORTED`. Neither cell is ever dropped from the dataset.

The checker itself is a **direct, renamed generalization** of `attester.py`'s
`_verify_scrub()`: same first-EXIT-boundary, same textual-order approach,
explicitly documented as a *structural-signature checker*, not a CFG
verifier (task §13). It gets a second capability `_verify_scrub()` lacked:
recognizing the intermediate `STORE_ONLY`/`PARTIAL_SIGNATURE` states instead
of a binary PASS/FAIL, needed because this study's constructs (Z2/Z3/Z4)
deliberately produce partial signatures that `attester.py` was never asked
to distinguish.

---

## 6. Non-circular ground truth

Three separate fields on every record, never conflated (task §15):

- `source_intent_label` — which Z construct was generated (e.g. `Z5`). Pure
  metadata about what was asked for.
- `binary_ground_truth` — assigned only by the stratified manual-review
  process in Phase 5, from raw SASS, independent of both
  `source_intent_label` and the automated checker.
- `checker_verdict` — the automated classifier's output from §5, evaluated
  for accuracy **only** against `binary_ground_truth`, never against
  `source_intent_label` (a compiler can silently remove or reorder a
  construct, so intent is not truth).

Stratified sample target: ≥200 artifacts (task's 200–300 floor for a
~2,000-artifact corpus), covering every construct, context, architecture,
toolkit, and outcome category, plus **all** `COMPILE_FAILED`,
`COMPILE_UNSUPPORTED`, and `UNDECIDABLE` cells without exception. Sampling
script and disagreement-resolution process specified in Phase 2's
`scripts/validate_dataset.py` design; actual manual annotation is Phase 5
work, not done here.

---

## 7. Dataset schema

JSONL, one record per attempted (construct, context, opt-level, arch,
toolkit) cell, fields exactly as enumerated in the task brief §16
(`artifact_id` through `notes` — ~50 fields). No fields added or removed
from that list; full type/nullability spec goes in `DATA_DICTIONARY.md`
(next Phase-2 deliverable, not duplicated here). `artifact_id` is
deterministic: `sha256(construct|context|opt|arch|toolkit)[:16]`, so re-runs
are idempotent and resumable (task §23).

---

## 8. Pilot matrix (recap, unchanged from Phase 0, now with rationale tied to §3's exclusion)

Z0, Z1, Z2, Z5, Z6 × K0, K2 × CUDA 12.8 × ptxas-O0/O3 × sm_86/sm_89 =
**5 × 2 × 1 × 2 × 2 = 40 artifacts**, plus **2 extra pilot-only cells**
(Z1×K0 and Z2×K3, both at ptxas-O3/sm_89/CUDA 12.8, each compiled twice —
once plain, once with `-maxrregcount=16 --use_fast_math`) specifically to
re-test the flag-exclusion decision in §3 against combinations the original
repo's sweep never covered. **42 pilot artifacts total.**

---

## 9. Feasibility compiles (run for real; this section now reports actual results)

Two combinations were picked as the highest-risk for a CUDA-validity mistake
in the design above. Sources, cubin, and raw SASS are preserved in
`dev_notes_phase1/` for audit (not deleted despite being throwaway probes —
they're now load-bearing evidence for a design decision below).

1. **Z6 × K4** (inline-PTX hardened scrub inside the tiled-GEMM context) —
   compiled clean, no warnings, WSL2 CUDA 12.8, `sm_89 -Xptxas -O3`. SASS
   confirms the expected `STS […], RZ` → `MEMBAR.SC.CTA` →
   `BAR.SYNC.DEFER_BLOCKING` ordering on `a_tile`, positioned before the
   divergent `if(row<M&&col<N)` output-write tail, exactly as designed.
2. **Z2 × K3** (volatile-no-barrier inside the reduction context) — also
   compiled clean. Disassembly showed something the design didn't
   anticipate: **the kernel's SASS contains four `EXIT` instructions** (three
   predicated — `@P1 EXIT`, `@!P0 EXIT` ×2 — plus one unconditional), even
   though the CUDA source has exactly one `return`-free exit point. This is
   ptxas's own code generation for the reduction tree's internal
   `if (tid < stride)` branching and per-warp instruction scheduling, not a
   second source-level exit. A raw static count of `STS […], RZ`
   instructions (29) also did not match the naive expectation of "256 words
   → 256 stores" — because `STS` is a per-warp SIMT instruction (one static
   instruction, executed once per warp reaching it, writing up to 32 words
   per execution), so static instruction count was never the right metric
   in the first place, and the 29 figure also includes an unrelated
   predicated zero-store from the earlier `(tid<n) ? … : 0.0f` fill
   ternary's false branch.

   Rather than resolve this by reasoning about the SASS text further, I
   built a diagnostic variant (`dev_notes_phase1/scratch_runtime_check.cu`)
   that actually **runs the kernel on the RTX 4050 (sm_89)** and has every
   thread read back its own post-scrub shared word through a global-memory
   dump, so completeness can be checked empirically instead of inferred.
   Result: `launch_err=no error`, `sync_err=no error`, reduction output
   matched the expected sum exactly, and **all 256 words read back as
   0.0f — zero threads skipped their scrub store.** The K3 template as
   designed is confirmed CUDA-correct on real hardware, not just
   plausible-looking source.

**Design consequence for the Phase-2 checker.** This is now measured, not
hypothetical, evidence that a *single-source-exit, barrier-safe* kernel can
still legitimately compile to **multiple SASS `EXIT` instructions**. The
original `attester.py`'s `_verify_scrub()` scopes its entire check to
"everything before the first `EXIT`" (task §13 already flagged this as a
known limitation; this probe shows a concrete case where that scoping would
have been the wrong boundary — the Z2×K3 scrub evidence here happens to sit
before the *first* predicated EXIT anyway in this particular instance, so
`_verify_scrub()` would not have been fooled on this specific artifact, but
the general pattern — legitimate scrub instructions appearing after an
early predicated EXIT belonging to a different warp's path — is now known
to occur in this corpus and cannot be assumed away). The new checker
(`classify_binary.py`, Phase 2) will therefore:
- Count all `EXIT` instructions per kernel and record `exit_count` on every
  artifact record (already in the schema, §7).
- For `exit_count == 1`: use the same "everything before EXIT" scoping as
  `attester.py`, unchanged.
- For `exit_count > 1`: do **not** silently assume first-EXIT scoping is
  correct or incorrect. Search the full kernel body for the required
  signature elements and their relative order; if the signature is found
  consistently relative to **every** EXIT the required elements should
  logically precede, classify normally, but flag the record
  (`recognized_ordering` notes `multi_exit_scope_assumed`); if ordering
  relative to multiple EXITs cannot be established with confidence, return
  `UNDECIDABLE` rather than guess — per task §13's explicit instruction that
  a shallow regex expansion must not be described as CFG analysis, and per
  §12's rule that `UNDECIDABLE` must never be silently coerced to a
  pass/fail verdict.
- This will be one of the specific things manually re-checked during the
  Phase-3 pilot's mandatory "manual inspection agrees with automated
  extraction" gate, using real pilot artifacts, not just this Phase-1 probe.

---

## 10. Stop conditions (task §25, restated as this study's operational checklist)

Any of the following halts progress and produces a report instead of a
guess: unsafe uncommitted repo changes; unresolvable CUDA tool paths;
`COMPILE_UNSUPPORTED` for a ptxas option this plan assumed was valid;
non-compiling source; a barrier-placement pattern that risks deadlock;
parser/manual-inspection disagreement; incomplete artifact metadata;
circular classifier ground truth; insufficient disk; a required tool
install; an unsupported paper claim; or any experiment that would just
reproduce the original repo's existing DSE/EI/overhead/tamper-ladder
evidence rather than generate new evidence.

---

## 11. Secondary sensitivity study and replication (task §17–18, design only, not run yet)

Deferred verbatim to the task's own spec until the core corpus (Phase 4) and
its validation (Phase 5–6) pass: constructs {Z1,Z2,Z5,Z6} × contexts
{K0,K2, one of K3–K5} × buffer sizes {256,1024,4096,+1 more} × block sizes
{32,64,128,256} × ptxas {O0,O3} × arch {sm_86,sm_89} = 768 artifacts.
Replication: ≥50 stratified configs compiled 3× each, comparing cubin/SASS
hashes and extracted features to check the determinism assumption honestly
rather than assert it.

---

## 12. Files this phase adds

Only `robustness_study/PLAN.md` (this file). `DATA_DICTIONARY.md`,
`configs/*.yaml`, and all `scripts/`/`kernels/`/`tests/` scaffolding are
Phase 2 deliverables, written after the §9 feasibility compiles confirm the
design above is sound.
