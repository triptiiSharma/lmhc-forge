# LMHC-Forge — Multi-GPU Replication Harness

This harness replicates the residue / LMHC-mitigation measurement from
`src/leakage_demo.cu` on whichever single NVIDIA GPU is available. Methodology
is held byte-for-byte identical to the existing controlled experiment so that
per-GPU numbers are directly comparable to the RTX 4050 baseline.

**If you are a collaborator**: follow [Quick Start](#quick-start) below, then
send back the one JSON file it produces.

> Reports are generated from validated JSONs and depict controlled-placement measurements only.

---

## Scientific Framing

> Residue is fully observable when the subsequent kernel is co-resident on the
> same SM — co-residency is forced in this experiment to demonstrate the window.
> This is in-process cross-kernel shared-memory persistence, NOT cross-tenant
> LeftoverLocals on NVIDIA.
>
> Co-residency is forced on every device, so these are controlled-placement
> results consistent with our adversarial threat model — not natural-scheduling
> frequencies, and not a claim about production prevalence or about non-tested
> hardware.

Read [`../docs/INTERPRETATION.md`](../docs/INTERPRETATION.md) for the pre-committed reading of every possible outcome
before any results are collected.

---

## Quick Start (collaborators — one command)

```bash
git clone <repo-url> lmhc-forge
cd lmhc-forge/multi_gpu
bash run.sh
```

Paths with spaces are supported. The script must be run as `bash run.sh` inside WSL2 (not PowerShell or cmd.exe directly).

`run.sh` auto-detects compute capability, compiles, runs all three phases
(control / residue / LMHC scrub), saves the JSON and a run log to
`runs/<timestamp>_<gpu>/`, and copies the JSON to `results/`. Send back
the JSON file whose path is printed at the end.

---

## Prerequisites

| Requirement | Notes |
|---|---|
| NVIDIA GPU (any SM 5.0 +) | Tested on Ada Lovelace (sm_89); any Kepler+ should work |
| CUDA Toolkit (nvcc) | 11.x or newer; 12.8 confirmed. See install notes below. |
| Host C compiler | gcc-14, gcc-12, or gcc-11 (nvcc 12.8 requires gcc ≤ 14) |
| nvidia-smi | For auto-detecting compute capability in run.sh |
| No Python required | The harness is pure C + CUDA; Python only needed for aggregate.py |

### Installing CUDA (Debian / Ubuntu / Kali)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y cuda-nvcc-12-8 gcc-14
```

On Kali Linux 2026.1 the apt repo may report a SHA1 signature error. Fix:

```bash
# In /etc/apt/sources.list.d/cuda-debian12-x86_64.list, change the repo line to:
deb [trusted=yes] https://developer.download.nvidia.com/compute/cuda/repos/debian12/x86_64/ /
```

### One-time glibc 2.42 patch (Kali 2026.1 / glibc ≥ 2.42 only)

glibc 2.42 added C23 math functions (`cospi`, `sinpi`, `rsqrt`, `cospif`,
`sinpif`, `rsqrtf`) that conflict with CUDA 12.8's declarations. If you see
errors like `declaration of 'cospi' with attribute 'nothrow' conflicts with
previous declaration`, apply this one-time patch:

```bash
MH=/usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h
sudo cp "$MH" "${MH}.orig"
# Add __THROW to the six conflicting declarations.
# The exact lines to patch are:
#   extern double cospi(double x);          → extern double cospi(double x) __THROW;
#   extern float  cospif(float x);          → extern float  cospif(float x) __THROW;
#   extern double sinpi(double x);          → extern double sinpi(double x) __THROW;
#   extern float  sinpif(float x);          → extern float  sinpif(float x) __THROW;
#   extern double rsqrt(double x);          → extern double rsqrt(double x) __THROW;
#   extern float  rsqrtf(float x);          → extern float  rsqrtf(float x) __THROW;
```

The original is preserved at `math_functions.h.orig`. This patch is in-WSL /
in-VM only; it has no effect on Windows-native CUDA.

---

## Manual Build and Run

If `run.sh` does not auto-detect your setup correctly:

```bash
# 1. Compile — substitute your arch (sm_75 for Turing, sm_86 for Ampere, etc.)
nvcc -arch=sm_89 -O2 -ccbin /usr/bin/gcc-14 \
     -o multi_gpu_repro multi_gpu_repro.cu

# 2. Run with defaults (N=200, 16 KB buffer — matches src/leakage_demo.cu)
./multi_gpu_repro -o results/MyGPU_sm89_driver.json

# Or pipe to stdout and rename:
./multi_gpu_repro > results/MyGPU_sm89_driver.json
```

CLI flags (all optional):

| Flag | Default | Meaning |
|---|---|---|
| `-n <trials>` | 200 | Trials per phase (matches `leakage_demo.cu`) |
| `-o <file>` | stdout | Write JSON to this file |

The harness prints a suggested output filename to stderr, e.g.:
`NVIDIA_GeForce_RTX_4090_sm89_535.161.json`

---

## What the Harness Measures

Three phases run in sequence on a single GPU:

| Phase | Kernel sequence | Measures |
|---|---|---|
| Residue | `kernelA_unscrubbed` → `kernelB_listener` | EI without scrub |
| Control | `kernelB_control` only (writer never launched) | Validity gate — must be 0.0 |
| LMHC scrub | `kernelA_scrubbed` → `kernelB_listener` | EI with hardened scrub |

**Validity gate**: if the control Exposure Index is non-zero, the JSON is marked
`valid: false`. Do not report a non-zero control result as a leak; it indicates
possible self-contamination or driver behavior outside the model.

**Forced co-residency**: `__launch_bounds__(32, 1)` + single-block launches
plus `%smid` verification per trial. `%smid` is a documented PTX special
register (PTX ISA §9.4); reading it slightly perturbs scheduling but the effect
is negligible for single-block single-warp kernels.

---

## Output JSON Schema

```jsonc
{
  "schema_version": "multi_gpu_repro_v1",
  "timestamp": "2026-07-01T12:00:00Z",
  "valid": true,
  "valid_reason": "control all-zero (self-contamination ruled out)",

  "environment": {
    "gpu_name": "NVIDIA GeForce RTX 4090",
    "compute_capability": "sm_89",
    "architecture_name": "Ada Lovelace",
    "driver_version": "535.161",
    "cuda_runtime_version": "12.3",
    "cuda_driver_api_version": "12.3",
    "nvcc_version_compiled_with": "12.8.0",
    "os": "Linux 6.1.0 x86_64",
    "wsl2": false,
    "gpu_sm_count": 128,
    "shared_mem_per_sm_bytes": 102400
  },

  "config": {
    "buffer_words": 4096,
    "buffer_bytes": 16384,
    "buffer_geometry": "fixed compile-time constant (NWORDS=4096); identical to src/leakage_demo.cu baseline",
    "pattern_sentinel": "0xCAFEBABE",
    "block_dim": 32,
    "trials_per_phase": 200,
    "launch_bounds": "__launch_bounds__(32, 1)",
    "co_residency_method": "single-block, same-SM forced via __launch_bounds__(32,1); %smid verified per trial"
  },

  "residue_run": {
    "same_sm_trials": 200,
    "nonzero_word_count_same_sm": 819200,
    "total_word_count_same_sm": 819200,
    "exposure_index": 1.000000
  },

  "control_run": {
    "trials": 200,
    "nonzero_word_count": 0,
    "total_word_count": 819200,
    "exposure_index": 0.000000,
    "control_clean": true
  },

  "lmhc_scrub_run": {
    "same_sm_trials": 200,
    "nonzero_word_count_same_sm": 0,
    "total_word_count_same_sm": 819200,
    "exposure_index": 0.000000
  },

  "framing": "...(mandatory framing text)...",
  "result_summary": "VALID: control clean; LMHC scrub drives EI to 0.0 on this device."
}
```

---

## Aggregating Results

Once you have collected JSON files from multiple devices:

```bash
# Place all JSON files in multi_gpu/results/
python3 aggregate.py

# Write Markdown to a file and TSV side-car:
python3 aggregate.py --output results_table.md --tsv results_table.tsv

# Also emit an HTML comparison table (valid=false rows highlighted):
python3 aggregate.py --html results_table.html
```

The aggregation script reads only the JSON files — no numbers are hardcoded.

For a per-device HTML report with an EI bar chart:

```bash
python3 report.py results/MyGPU_sm89_driver.json
# Writes MyGPU_sm89_driver_report.html (self-contained; degrades to CSS bars if matplotlib absent)
```

---

## Scope Constraints

- **Controlled placement only.** This harness does not have an unconstrained
  or natural-scheduling mode. Do not add one here; if needed, create a
  separate exploratory path clearly labeled as non-controlled.
- **No DSE characterization.** The 58-STS / ordering-differential finding
  (Module 3) is toolchain-specific to ptxas 12.8/13.3 and is out of scope.
  Collaborators may run different CUDA versions; only residue + scrub
  elimination are measured.
- **No "modern GPUs" / "all NVIDIA GPUs" language.** Refer only to the
  specific device names that appear in the results table.
- **Additive only.** This harness does not modify or re-run any existing
  single-GPU experiments or their result files.

---

## Files in this directory

| File | Purpose |
|---|---|
| `multi_gpu_repro.cu` | CUDA harness — build and run this |
| `run.sh` | One-shot build + run script for collaborators |
| `aggregate.py` | Reads `results/*.json`, emits Markdown / TSV / HTML comparison table |
| `report.py` | Per-device HTML report with EI bar chart; run against any single result JSON |
| `README.md` | This file |
| [`../docs/INTERPRETATION.md`](../docs/INTERPRETATION.md) | Pre-committed outcome readings (written before results) |
| `results/` | Per-device JSON files (place outputs here; also written by `run.sh`) |
| `runs/` | Timestamped run archives: JSON + `stderr.log` + `run_metadata.txt` per run |
