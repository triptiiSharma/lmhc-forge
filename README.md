# LMHC-Forge

Evaluation artifact for the paper **"Formalizing Local Memory Hygiene Contracts for GPU Confidential Computing"** (Tripti Sharma, Arshad Ali, Guruansh Singh). The artifact provides measured evidence on a real GPU replacing the paper's estimated claims around the LeftoverLocals / CVE-2023-4969 class of GPU shared-memory residue: specifically, the cross-kernel residue rate, per-kernel scrub overhead, compiler dead-store-elimination (DSE) behaviour, and the independence of the attestation verifier. All measurements are from an NVIDIA RTX 4050 Laptop GPU (Ada Lovelace, sm_89) running under WSL2.

---

## What this measures

| Paper claim | Paper's original (estimated) value | Measured artifact result | File / Figure |
|---|---|---|---|
| **§6.2** — Recovery rate: fraction of shared-memory words readable by a co-resident listener kernel | ">90% of words" (estimated) | **EI = 1.000 → 0.000** (N=200 trials, 200/200 same-SM): 100% without scrub; 0% with LMHC hardened scrub | `results/leakage_demo.json`; `figures/fig3_exposure_index` |
| **Table I / §6.1** — Scrub overhead for three kernel classes (reduction, GEMM, attention) | "1–5% estimated" (all six cells placeholder) | **Below timing noise floor** on RTX 4050: σ upper bounds < 1.08 µs (GEMM), < 1.26 µs (reduction), < 1.77 µs (attention); isolation trend +0.17 µs for attention (positive, unresolved at 1σ) | `results/benchmark.json`; `figures/fig4_benchmark` |
| **§4.4 / Theorem 1** — DSE hazard: naive zeroing eliminated by compiler; volatile+fence survives | Elimination claimed for naive scrub; survival claimed for volatile+fence | **Ordering differential** (not per-store elimination) on ptxas 12.8 and 13.3 at -O3 sm_89: STS counts equal (58 each); hardened uniquely emits `MEMBAR.SC.CTA` + extra `BAR.SYNC`; pure-dead allocation-granularity DSE confirmed (naive → 0 STS, volatile → 29 STS where buffer never read) | `results/dse_differential.json`; `sass_dumps/dse_test.sass`; `figures/fig2_dse_differential` |
| **§5.4** — Verifier independence: attester re-derives scrub presence from binary without trusting witness fields | "Independent re-derivation" (asserted, not demonstrated) | **T1–T6 tamper-test ladder**: T1–T5 correctly FAIL (hash mismatch, missing scrub in SASS, unaccounted kernel, missing source hash); T6 PASS (known limitation — generate-from-scratch requires binary-level taint to close) | `src/attester.py`; `src/tamper_tests.py`; `figures/fig5_tamper_panel` |

*Hardware caveat: all measurements on RTX 4050 Laptop GPU (Ada Lovelace, sm_89, WDDM/WSL2 passthrough). Relative overhead generalises qualitatively; absolute µs values do not transfer to H100 or other targets.*

---

## Key results (honest framing)

**Residue (Module 6):** Cross-kernel shared-memory residue rate is 100% without scrub (every word of a 16 KB buffer reads back the writer kernel's sentinel 0xCAFEBABE) and 0% with the LMHC hardened scrub. This is **in-process, same-SM, forced co-residency** — not a reproduction of cross-tenant LeftoverLocals on NVIDIA. Co-residency was forced via `__launch_bounds__(32,1)` and single-block launches; the experiment demonstrates that the residue window exists, not how often it arises naturally.

**DSE (Module 3):** Per-store dead-store elimination was **not observed** on ptxas 12.8 or 13.3 for the fill→read→scrub pattern. ptxas performs allocation-granularity DSE only (eliminates entire allocations never read, not individual stores). The empirical differential between naive and hardened scrub is therefore **ordering**, not store presence: the hardened version uniquely emits `MEMBAR.SC.CTA` (`__threadfence_block`) which provides the C++ happens-before guarantee that the naive version lacks. Theorem 1 stands as a permission-level argument under the C++ abstract machine — a conforming compiler is permitted to eliminate plain stores; the volatile qualifier makes them un-eliminable — even though that permission was not exercised by ptxas on this toolchain.

**Overhead (Module 7):** The LMHC scrub cost is **below the timing noise floor** (σ ≈ 1.1–1.8 µs) of CUDA-event batch measurement for single-block kernels on the RTX 4050. All three kernel classes are reported as upper bounds (< σ), not point estimates. Negative median overheads in the raw data are measurement artefacts (scheduling jitter), not speedups. A scrub-isolation run (empty vs. scrub-only kernel, 1 000 launches/batch, 80 reps) gives +0.17 µs for the attention kernel's 2 048-word scrub at 64 threads — positive direction but not statistically significant at 1σ.

**Verifier (Module 5):** T1–T5 tamper tests are correctly rejected. T6 (adversary runs the full witness-generation pipeline on a de-annotated source) produces a PASS — a **documented limitation**: source-level attestation cannot distinguish a legitimately annotated source from a de-annotated one when the adversary controls both the source file path and the witness. Closing T6 requires binary-level taint from SASS, which is future work.

---

## Hardware / platform

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace) |
| Architecture | sm_89 |
| Shared memory / SM | ~100 KB |
| Driver | 573.22 (CUDA max 12.8) |
| Build environment | WSL2 Kali Linux 2026.1, glibc 2.42 |
| CUDA toolkit | 12.8 (nvcc 12.8, ptxas 12.8) |
| Host compiler | gcc-14 |

Reference target in the paper is the H100 (228 KB shared/SM). All numbers in this artifact are from the RTX 4050 and labeled as such; do not compare absolute µs values across platforms.

---

## Prerequisites

The following must be satisfied inside WSL2 before running `run_all.sh`.

**1. Hardware** — NVIDIA GPU, sm_80 or newer. Change `ARCH` at the top of `run_all.sh` if your GPU differs.

**2. WSL2** — Tested on Kali Linux 2026.1 (glibc 2.42). Other Debian-based WSL2 distributions may work if glibc ≥ 2.38.

**3. CUDA 12.8 toolkit**

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
dpkg -i cuda-keyring_1.1-1_all.deb
apt-get update && apt-get install -y cuda-nvcc-12-8 cuda-cuobjdump-12-8
```

If apt reports SHA1 signature errors on Kali 2026.1, add `[trusted=yes]` to the repo line in `/etc/apt/sources.list.d/`.

**4. gcc-14**

```bash
apt-get install -y gcc-14
```

**5. glibc 2.42 / CUDA 12.8 compatibility patch (one-time, system-wide — do not skip)**

glibc 2.42 adds C23 math functions (`cospi`, `sinpi`, `rsqrt`, `cospif`, `sinpif`, `rsqrtf`) declared `noexcept(true)` that conflict with CUDA 12.8's plain declarations. Without this patch, every `.cu` file that includes `<cmath>` fails to compile.

```bash
MH=/usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h
cp $MH ${MH}.orig          # keep the original as a backup
```

Then open `$MH` in an editor and add `__THROW` to the six conflicting declarations:

```
cospi   cospif
sinpi   sinpif
rsqrt   rsqrtf
```

Each of these appears once in the file as a plain `extern double/float` declaration. Change:

```c
extern double cospi(double x);
```

to:

```c
extern double cospi(double x) __THROW;
```

(and likewise for the other five). The original is preserved at `${MH}.orig` for reference.

**6. Python 3 + matplotlib + numpy**

```bash
apt-get install -y python3-matplotlib python3-numpy
```

**7. cuobjdump** — installed by `cuda-cuobjdump-12-8` (step 3 above).

**Optional: nvcc 13.3 (for DSE comparison only)**

```bash
apt-get install -y cuda-nvcc-13-3 cuda-cuobjdump-13-3
```

Note: nvcc 13.3 requires `g++-14` as the host compiler (it emits C++ guard symbols). The main pipeline uses nvcc 12.8 only.

---

## Repository structure

```
lmhc-forge/
├── SPEC.md                        Source-of-truth build specification
├── RESULTS.md                     Claim-mapping: paper estimates → measured values
├── run_all.sh                     Full pipeline script (Phase 0 → all modules → figures)
├── research_paper.pdf             The paper this artifact accompanies
│
├── src/
│   ├── phase0_residue.cu          Phase 0: cross-launch residue detector
│   ├── phase0_control.cu          Phase 0 control: Kernel B only (no writer)
│   ├── discover.py                Module 1: __shared__ buffer discovery (conservative + taint)
│   ├── lmhc_rewriter.py           Module 2: AST rewriter inserting scrub blocks
│   ├── dse_test.cu                Module 3: four probe kernels for DSE differential
│   ├── dse_minimal.cu             Module 3: minimal repro
│   ├── dse_analysis.py            Module 3: parse cuobjdump SASS, emit dse_differential.json
│   ├── witness_gen.py             Module 4: compile + SHA-384(cubin ∥ policy) → witness.json
│   ├── attester.py                Module 5: independent verifier (3 checks, no witness trust)
│   ├── tamper_tests.py            Module 5: T1–T6 tamper-test suite
│   ├── leakage_demo.cu            Module 6: writer/listener EI measurement
│   ├── benchmark.cu               Module 7: interleaved batch benchmark + scrub isolation
│   ├── visualize.py               Generates all five figures (PNG 300 dpi + PDF)
│   ├── test_kernels.cu            Test kernels (attention_fwd, gemm_tile, reduction_sum)
│   ├── test_kernels_lmhc.cu       Module 2 output: test kernels with scrub blocks inserted
│   └── test_kernels_m4_lmhc.cu    Module 4 input: rewritten source used for witness
│
├── results/
│   ├── phase0_residue.json        Phase 0: residue rate 1.000 (N=1000)
│   ├── phase0_control.json        Phase 0 control: residue rate 0.000
│   ├── discovery_conservative.json  Module 1: all 5 __shared__ buffers flagged
│   ├── discovery_taint.json       Module 1: 2 confidential buffers (smem_qk, a_tile)
│   ├── dse_differential.json      Module 3: STS/MEMBAR/BAR.SYNC counts, verdict
│   ├── witness.json               Module 4: SHA-384 witness for test_kernels_lmhc
│   ├── test_kernels.cubin         Compiled cubin referenced by witness.json
│   ├── leakage_demo.json          Module 6: EI_before=1.0, EI_after=0.0
│   └── benchmark.json             Module 7: per-kernel overhead stats + isolation results
│
├── sass_dumps/
│   └── dse_test.sass              cuobjdump -sass output for Module 3 kernels
│
├── figures/
│   ├── fig1_phase0_heatmap.{png,pdf}      Phase 0 residue heatmap (64×64 word grid)
│   ├── fig2_dse_differential.{png,pdf}    Module 3 STS/MEMBAR/BAR.SYNC bar + SASS excerpt
│   ├── fig3_exposure_index.{png,pdf}      Module 6 EI before/after
│   ├── fig4_benchmark.{png,pdf}           Module 7 overhead upper bounds
│   └── fig5_tamper_panel.{png,pdf}        Module 5 T1–T6 PASS/FAIL matrix
│
└── multi_gpu/                             Multi-GPU replication harness (additive)
    ├── multi_gpu_repro.cu                 CUDA harness — byte-identical kernels to src/leakage_demo.cu
    ├── run.sh                             One-shot build + run for collaborators (bash run.sh)
    ├── aggregate.py                       Aggregate per-device JSONs → Markdown / TSV / HTML table
    ├── report.py                          Per-device HTML report with EI bar chart
    ├── INTERPRETATION.md                  Pre-committed outcome readings (written before results)
    ├── README.md                          Setup, schema, and scope documentation
    ├── results/                           Per-device JSON files (place collaborator outputs here)
    └── runs/                              Timestamped run archives (JSON + stderr.log + metadata)
```

---

## Reproducing

Clone the repo and run inside WSL2 after completing all prerequisites above:

```bash
# Clone into WSL2 filesystem (faster I/O than /mnt/c/)
git clone https://github.com/triptiiSharma/lmhc-forge.git /opt/lmhc-forge
cd /opt/lmhc-forge
bash run_all.sh
```

`run_all.sh` runs the following in order:

| Step | Module | What it does |
|------|--------|-------------|
| 1 | Phase 0 | Compiles and runs residue detector; writes `results/phase0_residue.json` |
| 2 | Module 1 | Runs `discover.py` on `test_kernels.cu`; writes `discovery_*.json` |
| 3 | Module 2 | Runs `lmhc_rewriter.py`; writes `test_kernels_lmhc.cu` |
| 4 | Module 3 | Compiles `dse_test.cu`, runs `cuobjdump -sass`, runs `dse_analysis.py`; writes `dse_differential.json` + `dse_test.sass` |
| 5 | Module 4 | Runs `witness_gen.py`; writes `witness.json` + `test_kernels.cubin` |
| 6 | Module 5 | Runs `attester.py` (T3 baseline) then `tamper_tests.py` (T1–T6) |
| 7 | Module 6 | Compiles and runs `leakage_demo`; writes `leakage_demo.json` |
| 8 | Module 7 | Compiles and runs `benchmark`; writes `benchmark.json` |
| 9 | Figures | Runs `visualize.py`; writes all 10 files in `figures/` |
| 10 | Check | Non-determinism check on leakage_demo (EI values must match across two runs) |

Expected wall time: approximately 5–10 minutes (dominated by GPU warm-up and the 80-rep × 1 000-launch scrub-isolation runs in Module 7).

---

## Figures

| File | Shows |
|------|-------|
| `fig1_phase0_heatmap` | 64×64 word grid of shared-memory words before and after LMHC scrub; EI 1.0 (all words = 0xCAFEBABE) vs. EI 0.0 (all words zero) |
| `fig2_dse_differential` | Bar chart of STS / MEMBAR.SC.CTA / BAR.SYNC counts for scrub_naive, scrub_hardened, scrub_ptx_fallback; annotated SASS excerpt showing absent/present MEMBAR |
| `fig3_exposure_index` | Two-bar chart of Exposure Index before (1.000) and after (0.000) LMHC scrub; Module 6 framing caption |
| `fig4_benchmark` | Baseline latency with σ upper bounds for LMHC overhead annotated per kernel class; right panel shows overhead as % of baseline (all marked "< σ bound") |
| `fig5_tamper_panel` | T1–T6 matrix: test description, CHECK 1/2/3 outcome, verdict; T6 highlighted as known limitation |

All figures: 300-dpi PNG + vector PDF, Okabe-Ito colorblind-safe palette, hardware and trial count in every caption.

---

## Multi-GPU Replication

The `multi_gpu/` directory contains an additive harness that replicates the
residue / LMHC-mitigation measurement from `src/leakage_demo.cu` on any NVIDIA
GPU a collaborator can access. It does **not** modify any single-GPU experiment,
result file, or figure.

Kernels are byte-identical to `src/leakage_demo.cu` (same `__shared__` geometry,
same sentinel, same `__launch_bounds__(32,1)` forced co-residency, same N=200
default). The only addition is `%smid` verification per trial and per-device JSON
output.

**One-command collaborator flow (inside WSL2):**

```bash
cd multi_gpu
bash run.sh    # auto-detects arch, compiles, runs 3 phases, saves JSON + run log
```

**Aggregating results from multiple devices:**

```bash
python3 multi_gpu/aggregate.py
python3 multi_gpu/aggregate.py --html multi_gpu_results.html
```

See `multi_gpu/README.md` for full setup, JSON schema, and scope constraints.
See `multi_gpu/INTERPRETATION.md` for pre-committed outcome readings written
before any collaborator results were collected.

> These results are controlled-placement measurements (forced co-residency on
> every device). No result from this harness is a claim about natural-scheduling
> frequencies or about devices not listed in the results table.

---

## Limitations

- **Single-block / single-exit scope.** `benchmark.cu` kernels use a single grid block. `lmhc_rewriter.py` handles single-exit kernels only; multi-exit kernels are rejected with an error. These are prototype limitations, not fundamental ones.
- **RTX 4050 vs. H100.** All timing and occupancy numbers are from the RTX 4050 (sm_89, ~100 KB shared/SM). The paper's reference target is the H100 (sm_90, 228 KB/SM). Relative overhead and qualitative DSE findings generalise; absolute µs values do not.
- **In-process vs. cross-tenant.** The residue measurement is intra-process (same CUDA context, consecutive kernel launches with forced co-residency). It validates the residue window LMHC is designed to close. It does not reproduce cross-tenant LeftoverLocals; NVIDIA was not affected cross-process per the paper.
- **T6 verifier boundary.** The generate-from-scratch attack (T6) passes all three verifier checks. Closing it requires binary-level taint re-derivation from SASS — tracing which shared-memory writes carry data from pointer arguments independently of source annotations. This is future work.
- **Source-annotation-level taint.** Module 1's taint pass is syntactic (intra-procedural, `__confidential__`-annotation-driven). It will miss confidential data introduced via pointer aliasing or function calls not visible in the kernel body.
- **glibc 2.42 patch.** The one-time patch to `math_functions.h` is a local system-file change with no upstream fix in CUDA 12.8. The original is preserved at `math_functions.h.orig`. Upstream fix: upgrade to CUDA 13.x once driver compatibility permits.

---

## Citation

```bibtex
@inproceedings{sharma2025lmhc,
  author    = {Tripti Sharma and Arshad Ali and Guruansh Singh},
  title     = {Formalizing Local Memory Hygiene Contracts for {GPU} Confidential Computing},
  booktitle = {TBD},
  year      = {TBD},
}
```
