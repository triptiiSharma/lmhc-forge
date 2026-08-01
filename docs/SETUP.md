# LMHC-Forge — Setup & Reproduction

Full prerequisites, environment patches, and reproduction steps. For the
short version, see the [Quickstart](../README.md#quickstart) in the main
README.

## Hardware / platform

| Item | Value |
|------|-------|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace) — primary target; see [`docs/RESULTS.md`](RESULTS.md) Multi-GPU Replication for RTX 3050 Laptop GPU and RTX 3060 Ti (Ampere) results |
| Architecture | sm_89 (primary); sm_86 (Ampere devices) |
| Shared memory / SM | ~100 KB |
| Driver | 573.22 (CUDA max 12.8) on the primary RTX 4050 dev machine |
| Build environment | WSL2 Kali Linux 2026.1, glibc 2.42 |
| CUDA toolkit | 12.8 (nvcc 12.8, ptxas 12.8); nvcc 13.3 optional, DSE comparison only |
| Host compiler | gcc-14 |

Reference target in the paper is the H100 (228 KB shared/SM). All numbers in
this artifact are from consumer GPUs and labeled as such; do not compare
absolute µs/ns/W values across platforms without re-measuring.

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

**Optional: clang/LLVM (for the IR-level DSE-possibility exhibit only)**

```bash
apt-get install -y clang llvm
```

Only needed to reproduce `src/dse_ir_build.sh` (a generic C/LLVM analog of the
DSE hazard, independent of nvcc's closed-source device compiler — see
[`docs/RESULTS.md`](RESULTS.md) "DSE-possibility exhibit").

## Repository structure

```
lmhc-forge/
├── README.md                      Short pitch, quickstart, links to docs/
├── LICENSE                        MIT
├── run_all.sh                     Full pipeline script (Phase 0 → all modules → figures)
│
├── src/                           CUDA + Python source for every module and Phase-1 addition
├── results/                       Measured outputs (json + csv)
├── figures/                       Generated figures (PNG 300dpi + PDF)
├── sass_dumps/                    cuobjdump -sass / LLVM IR dumps referenced by docs/RESULTS.md
│
├── docs/
│   ├── SPEC.md                    Build specification (source of truth for scope/claims)
│   ├── RESULTS.md                 Claim-mapping: paper estimates → measured values (full detail)
│   ├── SETUP.md                   This file
│   ├── INTERPRETATION.md          Multi-GPU pre-committed outcome readings
│   ├── research_paper.pdf         Original accompanying paper
│   └── LMHC_v2.docx               Current/expanded paper draft
│
└── multi_gpu/                     Multi-GPU replication harness (additive; see multi_gpu/README.md)
```

See [`docs/RESULTS.md`](RESULTS.md) for the full file-by-file mapping of every
module's inputs/outputs, including everything added after the original
Modules 1–7 build (P-spill/P-stage residue paths, on-device `clock64()`
timing, the analytical cost model, verifier-cost instrumentation, unforced
co-residency, and power sampling).

## Reproducing the original Modules 1–7 pipeline

Clone the repo and run inside WSL2 after completing all prerequisites above:

```bash
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
| 10 | Check | Non-determinism check on `leakage_demo` (EI values must match across two runs) |

Expected wall time: approximately 5–10 minutes.

## Reproducing the Phase-1 additions

These are not yet wired into `run_all.sh` (each is a standalone script/binary
documented in [`docs/RESULTS.md`](RESULTS.md)); run them individually from
the repo root inside WSL2:

| What | Build & run |
|---|---|
| P-spill / P-stage residue | `nvcc -arch=sm_89 -O3 -ccbin gcc-14 -o /tmp/pspill src/leakage_demo_pspill.cu && /tmp/pspill > results/leakage_demo_pspill.json` (same pattern for `leakage_demo_pstage.cu`) |
| Extended ptxas flag sweep | `bash src/dse_attempts.sh` |
| IR-level DSE-possibility exhibit | `bash src/dse_ir_build.sh` (requires clang/llvm, see Prerequisites) |
| Unforced co-residency | `nvcc -arch=sm_89 -O3 -ccbin gcc-14 -o /tmp/unforced src/unforced_coresidency.cu && /tmp/unforced > results/unforced_coresidency.json` |
| On-device `clock64()` overhead | `nvcc -arch=sm_89 -O3 -ccbin gcc-14 -lm -o /tmp/bc64 src/benchmark_clock64.cu && /tmp/bc64 > results/benchmark_clock64.json` |
| Analytical model validation | `python3 src/model_predict.py` (reads `results/benchmark_clock64.json`) |
| Verifier cost | `python3 src/verifier_perf.py` |
| Power sampling | `bash src/power_measure.sh` (requires `nvidia-smi --query-gpu=power.draw` support) |

## Multi-GPU replication (collaborator flow)

See [`multi_gpu/README.md`](../multi_gpu/README.md) for the full one-command
collaborator flow (`bash multi_gpu/run.sh`) and
[`docs/INTERPRETATION.md`](INTERPRETATION.md) for the pre-committed reading
of every possible outcome.
