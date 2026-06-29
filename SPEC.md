# LMHC-Forge — Build Specification

This is the source-of-truth spec for the LMHC-Forge evaluation artifact. If you
(Claude Code) lose context during the build, re-read THIS file rather than asking
for the plan to be re-explained.

You are helping build LMHC-Forge, the evaluation artifact for a research paper
titled "Formalizing Local Memory Hygiene Contracts for GPU Confidential Computing."
The paper PDF is in this repo (research_paper.pdf). Read it first — the artifact
must support its specific claims, not a generic version of them.

## Hardware

NVIDIA RTX 4050 (Ada Lovelace, ~100 KB shared memory/SM, configurable). The paper's
reference target is the H100 (228 KB/SM). Every number produced here comes from the
4050 and MUST be labeled as such. Relative overhead generalizes; absolute laptop
numbers do not.

## The three quantitative deliverables (the whole point of the artifact)

The artifact exists to replace claims the paper currently states without measurement.
There are THREE such claims — all first-class, equal priority:

1. **Recovery rate** — measured word-recovery percentage from a real writer/listener
   experiment on the 4050. Replaces the estimated ">90% of words" in §6.2.

2. **Overhead table** — real per-kernel scrub overhead for three kernel classes
   (memory-bound, GEMM, attention-style). Replaces all six estimated cells in Table I
   and the blanket §6.1 "replace with measurements" flag.

3. **DSE differential** — actual SASS showing the naive zeroing loop is compiled away
   (dead store eliminated) while the volatile+fence version survives in the emitted
   binary. This is the ONLY evidence that makes Theorem 1 / §4.4 non-circular: the
   paper asserts the hazard exists and the fix prevents it but never shows the compiler
   output proving both halves. Treat this with the same weight as #1 and #2 — it is
   NOT optional scaffolding.

## Module 6 framing — MANDATORY, use verbatim in all outputs and captions

"Residue is fully observable when the subsequent kernel is co-resident on the
same SM — co-residency is forced in this experiment to demonstrate the window.
This is in-process cross-kernel shared-memory persistence, NOT cross-tenant
LeftoverLocals on NVIDIA."

This distinction must appear:
- In every figure caption that shows the Exposure Index
- In the RESULTS.md mapping for deliverable #1
- In the README for Module 6
- In any inline comment in the leakage-demo source

## Non-negotiable honesty constraints (a reviewer will attack violations)

- Do NOT claim "NVIDIA zeroes shared memory since Ampere" or any undocumented
  mechanism. State only what we measure on THIS hardware, e.g. "across N trials on the
  RTX 4050 we observed no cross-kernel shared-memory residue."
- The leakage demo is NOT "reproducing LeftoverLocals on NVIDIA." NVIDIA was not
  affected per the paper. Frame it as "validating that the hardened scrub closes the
  in-process residue window on accessible hardware."
- Threat-model realism is anchored on AMD/Apple/Qualcomm (where LeftoverLocals was
  confirmed) and on intra-VM multi-kernel pipelines — never on NVIDIA's behavior.
- Every figure carries the hardware/platform caveat and trial count.

## Build order — STOP at the Phase 0 gate and report before continuing

### PHASE 0 — Residue Detector (GATE)  [COMPLETE — 2026-06-25]

Two separate kernel launches in one process, same buffer size. Kernel A writes a known
pattern to __shared__ and exits; Kernel B declares an identically-sized __shared__
buffer and reads it BEFORE writing. Force reuse: low occupancy, __launch_bounds__,
single block, pin to one SM if possible. Run N=1000 trials. Output: fraction of words
that are non-zero (the "residue rate"). A kernel reading its OWN uninitialized buffer
proves nothing — it must be cross-launch.

**Measured result (RTX 4050, N=1000):**
- Kernel A then Kernel B: residue rate = 1.000000, pattern 0xCAFEBABE in 4096000/4096000 words
- Control (Kernel B only, no Kernel A in process): residue rate = 0.000000, 0xCAFEBABE in 0/4096000 words
- 0xCAFEBABE appears if and only if Kernel A ran first in the same process.

Apply Module 6 framing verbatim to every description of these numbers.
Results saved to results/phase0_residue.json and results/phase0_control.json.

### MODULE 1 — Discovery Engine
Identify sensitive __shared__ buffers. Ship the conservative version first: flag EVERY
__shared__ buffer (matches the paper's over-approximation, Definition 1). Add taint
refinement (forward-propagate from __confidential-annotated params) as a second pass
that narrows the set.

### MODULE 2 — LMHC Rewriter
Takes a .cu file, inserts before every exit, for every flagged buffer: a
volatile-qualified cooperative zeroing loop + __threadfence_block() + __syncthreads().
Use libclang for a real AST transform handling multiple exits and early returns. If
libclang is impractical, scope explicitly to single-exit / structured-return kernels
and STATE that limitation in the README — do not ship a regex hack under a
"compiler-enforced" title.

### MODULE 3 — DSE Differential (delivers quantitative deliverable #3)
Compile two versions of the same scrub: (a) naive `for(...) buf[i]=0;` and (b) the
hardened volatile/fenced version. Run `cuobjdump -sass` on both at -O3. Show naive
shared-memory stores are ELIMINATED and hardened ones SURVIVE. Output a machine-checked
count of shared-store (STS / st.shared) instructions in each. If volatile+fence does
not survive on Ada, try the PTX st.shared inline-asm fallback the paper mentions and
re-dump.

### MODULE 4 — Witness Generator
Emit witness.json: kernel_hash = SHA-384(compiled cubin || policy) — hash the BINARY
not source — plus sensitive_regions, scrub_sites, policy_version, all_exits_protected.

### MODULE 5 — Attestation Verifier (must be INDEPENDENT)
Takes the .cubin + witness.json and RE-DERIVES the property from the binary:
disassemble, confirm scrub instructions dominate every exit, recompute the hash,
compare. Print PASS/FAIL. It must NOT simply re-read what Module 4 asserted — that
circularity is the thing §5.4 argues against. Must FAIL on a tampered binary whose
witness still claims all_exits_protected: true.

### MODULE 7 — Benchmark Suite (delivers quantitative deliverable #2)
GEMM, attention-style, reduction kernels. Time with CUDA events: warmup runs,
>=100 iterations, kernel-only timing, median + stddev. Report BOTH absolute scrub time
(µs) and relative % overhead. Note occupancy if the attention kernel's shared footprint
limits it on the 4050.

### MODULE 6 — Leakage Demo (delivers quantitative deliverable #1)
Writer leaves a known pattern; listener reads it; with LMHC applied, listener reads
zeros. Produces the Exposure Index before/after. Use the Phase 0 measured behavior to
frame this correctly (see honesty constraints).

## Visualizations

matplotlib; save each as 300-dpi PNG AND vector PDF for IEEE figures; consistent style,
colorblind-safe palette, captions stating hardware + N.
- Phase 0: residue heatmap — buffer words as a grid, colored by value, before any scrub.
- Module 3: side-by-side bar of shared-store instruction counts (naive vs hardened),
  plus an annotated SASS snippet image showing the missing/present stores.
- Module 6: Exposure Index bar chart — non-zero-word fraction before vs after LMHC.
- Module 7: grouped bar chart — baseline vs scrubbed per kernel class, both absolute µs
  and relative %; error bars from stddev.
- Module 5: a PASS/FAIL verification panel (per kernel, hash match, exits covered).

## LIMITATIONS (mandatory section in final README and RESULTS.md)

### Toolchain
- All CUDA compilation runs in WSL2 Kali Linux 2026.1 (glibc 2.42) using
  CUDA 12.8 toolkit + gcc-14. The Windows CUDA 13.3 toolkit cannot initialize
  on driver 573.22 (cudaErrorInsufficientDriver, error 35).
- A one-time patch was applied to
  `/usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h`
  to add `__THROW` to six C23 math function declarations (cospi, sinpi, rsqrt,
  cospif, sinpif, rsqrtf) that conflict with glibc 2.42's noexcept(true)
  declarations. Original preserved at math_functions.h.orig. Upstream fix:
  update NVIDIA driver to support CUDA 13.x or downgrade system glibc.
- Results are labeled RTX 4050 Laptop GPU (Ada Lovelace, sm_89, WDDM/WSL2
  passthrough). They generalize qualitatively; absolute timing does not.

### Phase 0 co-residency
- Diff-SM was 0/1000 because single-block launches with low occupancy
  (__launch_bounds__(32,1)) almost always land on the same SM back-to-back.
  Co-residency is forced, not a statistical natural occurrence.
  Framing: the experiment demonstrates THAT the window exists, not HOW OFTEN
  it arises in production pipelines (which depends on scheduling policy).

### Module 2 rewriter scope
- Scoped to kernels with a single exit point (no early returns, no goto).
  Kernels with multiple returns are rejected with an explicit error.
  This is a prototype limitation, not a fundamental one; a full libclang
  AST transform would handle arbitrary control flow.

## Deliverables

- Clean repo: /src (cuda + python), /figures, /results (json + csv), /sass_dumps, README.
- A single `run_all.sh` (or `make all`) that runs every module and regenerates all figures.
- RESULTS.md mapping each replaced paper claim:
  Table I estimates -> Module 7 measured; ">90% recovery" -> Module 6 measured;
  Theorem 1 / §4.4 -> Module 3 SASS differential; "verifier checks witness" -> Module 5
  re-derivation.
- LIMITATIONS section: single-exit scope (if applicable), 4050 vs H100, in-process vs
  cross-tenant, N trials, and any honesty caveat above.

Ask targeted questions ONLY if something is genuinely ambiguous before starting.
Otherwise build Phase 0 completely, stop at the gate, and report with real output.

---

## POST-BUILD EXTENSION — Multi-GPU Replication Harness [COMPLETE — 2026-06-29]

`multi_gpu/` was added after all modules above were verified. It is **additive
only** — kernels, result files, and figures from Modules 1–7 / Phase 0 were not
modified.

**What it adds:**
- `multi_gpu_repro.cu`: Unified CUDA harness (residue + control + LMHC scrub),
  kernels byte-identical to `src/leakage_demo.cu` (static `__shared__ uint32_t
  buf[NWORDS]`, same geometry, same `__launch_bounds__(32,1)`, same sentinel).
  `%smid` is read per trial for co-residency verification (PTX ISA §9.4;
  slightly perturbs scheduling, documented). Control runs FIRST (before any
  writer kernel) to mirror the phase0_control.cu separation.
- `run.sh`: One-shot build + run for collaborators. Handles paths with spaces,
  auto-detects compute capability, writes timestamped run archives under
  `runs/<TS>_<gpu>/` and copies JSON to `results/`.
- `aggregate.py`: Reads `results/*.json`, emits Markdown / TSV / HTML table.
- `report.py`: Per-device HTML report with EI bar chart (matplotlib or CSS fallback).
- `INTERPRETATION.md`: Pre-committed outcome readings written before any
  collaborator results were collected.

**Verified on RTX 4050 Laptop GPU (sm_89, WSL2):**
- EI_control = 0.0000 (validity gate passes)
- EI_residue = 1.0000 (100% — matches `results/leakage_demo.json`)
- EI_lmhc_scrub = 0.0000

**Scope constraints (hard):** controlled-placement only; no natural-scheduling
mode; no DSE characterization; no "modern GPUs" / "all NVIDIA GPUs" language;
refer only to specific device names in the results table.
