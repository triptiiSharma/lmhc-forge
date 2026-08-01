# AUDIT.md — UPES Robustness Study, Phase 0

Status: **Phase 0 complete.** This document is a factual inventory only. No
experiment has been implemented or run beyond a single throwaway toolchain
probe (compiled to `/tmp/audit_probe` inside WSL2, not committed, not part of
the repo).

Date of audit: 2026-08-01. Branch: `upes-robustness-study`, created from
clean `main` at commit `25a8f06`.

---

## 1. Repository state

```
$ git status        → clean, nothing to commit
$ git branch --show-current → main (before this branch was created)
$ git rev-parse HEAD → 25a8f068d675a61f1cc62a15f26e0271fec55aed
```

Recent history:
```
25a8f06 Add Phase-1 experiments, multi-GPU results, figures, and reorganize docs
138538c Add multi_gpu/ replication harness and update project documentation
5816b7f Add professional README
e992843 Add LMHC-Forge evaluation artifact (Modules 1-7, T1-T6, all figures)
```

The repository was clean, so per the task's safety rules the new branch
`upes-robustness-study` was created directly from `main`, and this audit was
written under a new `robustness_study/` directory. Nothing under `results/`,
`figures/`, `sass_dumps/`, `src/`, or `run_all.sh` was touched.

A **second, non-git working copy** of this project exists inside WSL2 at
`/opt/lmhc-forge` (created 2026-06-25, no `.git` directory). This appears to
be the native-filesystem copy used to actually execute `run_all.sh` and
produce the results later committed to the Windows-side repo (compiling
across the 9p `/mnt/c` bridge is slower and `run_all.sh` is hardcoded to
`/opt/lmhc-forge`). **It was not modified and will not be used** — the
Windows repo at `C:\Tripti Sharma\lmhc-forge` (== `/mnt/c/Tripti
Sharma/lmhc-forge` in WSL2) is the canonical, version-controlled copy, and
all new study work happens there, under `robustness_study/`.

---

## 2. Existing reusable assets (inspected in full)

| File | Purpose | Reusable logic for this study |
|---|---|---|
| `src/dse_test.cu` | 3 kernels: `scrub_naive`, `scrub_hardened`, `scrub_ptx_fallback`, all fill→use→scrub (K2-shaped), single exit, `DSE_NWORDS=1024`, `DSE_BLOCK=32` | Template basis for Z1/Z5/Z6 constructs and K2 context |
| `src/dse_minimal.cu` | 4 probe kernels: `nosync_naive/volatile` (K1-shaped, no pre-scrub barrier), `pure_dead_naive/volatile` (K0-shaped, buffer never read) | Template basis for K0/K1 contexts, Z1/Z2 constructs |
| `src/dse_analysis.py` | Parses `cuobjdump -sass` text into per-function blocks; counts `STS`, `MEMBAR`, `BAR.SYNC` via regex; **hardcoded to the 3 `KERNEL_NAMES` above** | `parse_sass()`'s function-splitting logic and Itanium-length-prefix `demangle_simple()` generalize directly; the report/verdict logic is specific to the old 3-kernel comparison and will not be reused as-is |
| `src/attester.py` | 3-check verifier: (1) SHA-384(cubin\|\|policy) hash match, (2) `_verify_scrub()` — regex-based `STS…RZ → MEMBAR.SC.CTA → BAR.SYNC` ordering before **the first `EXIT`**, (3) kernel-inventory taint completeness via `discover.py` | `_parse_sass()` (same function-splitter) and the instruction regexes (`_STS_RZ_RE`, `_MEMBAR_RE`, `_BAR_RE`, `_EXIT_RE`) are reusable primitives. `_verify_scrub()` itself is the thing the task requires me to *rename and frame honestly*, not reuse as a CFG verifier — see §12 below. |
| `src/discover.py` | Regex/pattern-based taint pass over `__shared__` buffers using a `__confidential__` marker macro | Not used by the robustness study (no confidentiality-taint dimension here); only referenced because `attester.py` imports it for CHECK 3, which the new study does not perform |
| `src/test_kernels.cu` | 3 realistic kernels (`attention_fwd`, `reduction_sum`, `gemm_tile`), single exit each | Template basis for K3 (reduction), K4 (GEMM), K5 (attention) contexts |
| `src/test_kernels_lmhc.cu`, `src/test_kernels_m4_lmhc.cu` | Rewriter output of the above with LMHC scrub inserted (Module 2/4 artifacts) | Reference for what a "hardened" rewrite of K3–K5 looks like; not executed directly |
| `run_all.sh` | Full original-artifact pipeline driver, hardcoded to `/usr/local/cuda-12.8`, `gcc-14`, `sm_89`, single toolchain | Not modified, not called. New study has its own driver scripts under `robustness_study/scripts/`. |
| `results/dse_differential.json`, `sass_dumps/dse_test.sass` | Original Module 3 measured evidence (see §3) | Read-only reference; preserved verbatim, never overwritten |
| `docs/SPEC.md`, `docs/RESULTS.md` | Build spec and full claim-mapping for the *original* paper/artifact | Read for context; establishes what must NOT be re-claimed as new evidence |

## 3. Existing measured evidence that MUST be preserved, not repeated

From `docs/RESULTS.md` (Module 3 section) and `results/dse_differential.json`,
confirmed by reading both files directly:

- **Finding (a)** — ptxas performs **allocation-granularity** DSE only: a
  `__shared__` buffer's stores are eliminated in their entirety only when the
  buffer has *zero reads anywhere in the kernel* (`pure_dead_naive` → 0 STS,
  `#550-D` warning). This is not per-store elimination.
- **Finding (b)** — the realistic fill→use→scrub pattern (`dse_test.cu`,
  K2-shaped) retains scrub stores on **both** naive and hardened versions, on
  **both** ptxas 12.8 (V12.8.93) and ptxas 13.3 (V13.3.33), at `-O3`, on
  sm_89: 58 STS in each, byte-identical store counts.
- **Finding (c)** — `pure_dead_volatile` (29 STS) vs `pure_dead_naive` (0
  STS) on both ptxas versions: volatile prevents the whole-allocation
  elimination that fires on the naive version when the buffer truly has no
  reads.
- **Finding (d)** — the machine-checked differential for the realistic
  pattern is **ordering, not presence**: `scrub_hardened`/`scrub_ptx_fallback`
  uniquely contain 1 `MEMBAR.SC.CTA` + 3 `BAR.SYNC` vs. `scrub_naive`'s 0
  `MEMBAR` + 2 `BAR.SYNC`.
- **Extended flag sweep** (already run, sm_89 only, `scrub_naive`,
  ptxas 12.8): `-Xptxas -O3,--allow-expensive-optimizations=true`,
  `-maxrregcount=16`, `-maxrregcount=8`, `--use_fast_math`, and `-O2` were
  all tried — STS count stayed at 58 in every case. **No configuration in
  that sweep changed the outcome for that construct.**
- A generic LLVM/clang `-O2` exhibit (`src/dse_ir_exhibit.c`, not CUDA)
  confirms the *semantic possibility* of per-store DSE removing a naive
  scrub while a volatile one survives — offered as corroborating evidence
  that the hazard is real, explicitly not as a claim about ptxas internals.
- README/RESULTS.md caveat, confirmed independently below: **"The Windows
  CUDA 13.3 toolkit cannot initialize on driver 573.22
  (cudaErrorInsufficientDriver, error 35)."** This refers to *runtime*
  kernel execution on the natively-installed Windows CUDA 13.3, not
  compilation. It does not block compile-only (`-cubin`) use of Windows
  13.3, though a separate, more basic blocker exists there — see §5.

**Implication for this study**: the new corpus must not re-run
`dse_test.cu`/`dse_minimal.cu` unchanged and re-report the same numbers as
new findings. New Z-constructs (Z0, Z3, Z4) and new K-contexts (K3–K5) that
were *not* in the original sweep are where new evidence can legitimately
come from, plus the additional architecture-target and cross-toolkit axes
the original study did not explore at this scale (original Module 3 was
sm_89-only across both toolkits).

---

## 4. Environment inventory

### 4.1 Host OS / shell actually in use

`uname -a` (Git Bash on the Windows host):
```
MINGW64_NT-10.0-26200 Tripti 3.4.10-2e2ef940.x86_64 2024-07-09 21:35 UTC x86_64 Msys
```
This is **native Windows**, not the WSL2 Kali environment the README and
`run_all.sh` assume. This is a material deviation from the original setup
and is the first thing that had to be resolved before any compilation could
be planned.

### 4.2 CPU / RAM / disk (Windows host)

- CPU: Intel Core i7-12650H, 10 physical cores / 16 logical processors
  (hybrid P+E core design; `wmic cpu get NumberOfCores,NumberOfLogicalProcessors`).
- RAM: `wmic OS get FreePhysicalMemory,TotalVisibleMemorySize` → ~16.49 GB
  total, ~5.3 GB free at audit time (Windows-side; excludes WSL2's own VM).
- Disk (`C:`): 463 GB total, **114 GB free**.

### 4.3 WSL2 — the environment this study will actually use

`wsl.exe -l -v` showed one distro, `kali-linux`, state **Stopped** by
default (booted successfully in ~5s for this audit; no persistent daemon
was left running beyond the audit probe).

Inside `kali-linux`:
```
$ uname -a
Linux Tripti 6.6.87.2-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC Thu Jun 5 18:30:46 UTC 2025 x86_64 GNU/Linux
$ ldd --version
ldd (Debian GLIBC 2.42-16) 2.42
```

- CPU: `nproc` → 16 (WSL2 exposes all logical processors from the host).
- RAM: `free -h` → **7.6 GiB total**, ~7.0 GiB free at audit time. This is a
  real, binding constraint (WSL2's memory ceiling, presumably set by
  `.wslconfig` or default 50%-of-host behavior) — materially smaller than
  the 16 GB visible to Windows.
- Disk: `df -h /` → **952 GB free** on the WSL2 native ext4 filesystem
  (`/dev/sdd`, 1007 GB total). `df -h /mnt/c` → **114 GB free** (same
  Windows `C:` drive, seen through the 9p bridge — this is where the actual
  git repo lives, so 114 GB, not 952 GB, is the effective ceiling for
  anything written into `robustness_study/`).
- Python: `python3 --version` → 3.13.14, with `matplotlib 3.10.7+dfsg1` and
  `numpy 2.3.5` already importable — confirmed by direct `import`.
- GPU passthrough: `nvidia-smi` inside WSL2 works and shows:
  ```
  NVIDIA-SMI 570.151   Driver Version: 573.22   CUDA Version: 12.8
  GPU 0: NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, sm_89), 6141 MiB, WDDM passthrough
  ```
  This is the **only GPU present**, and it is Ada Lovelace / **sm_89**. No
  other architecture has matching local hardware — confirmed, not assumed.

### 4.4 CUDA toolkits found

Two full, independent, working toolkit installs exist **inside WSL2**, both
already used to produce the committed results (confirmed via
`docs/RESULTS.md`: "Module 3 DSE comparison also run with nvcc/ptxas
13.3.33"):

| | CUDA 12.8 | CUDA 13.3 |
|---|---|---|
| Path | `/usr/local/cuda-12.8/` | `/usr/local/cuda-13.3/` |
| `nvcc --version` | release 12.8, V12.8.93, build `35583870_0` | release 13.3, V13.3.33, build `37862127_0` |
| `ptxas --version` | matches nvcc (bundled) | matches nvcc (bundled) |
| `cuobjdump --version` | release 12.8, V12.8.90 | release 13.3, V13.3.29 |
| `nvcc --list-gpu-arch` | compute_50,52,53,60,61,62,70,72,75,80,86,87,89,90,100,101,120 | compute_75,80,86,87,88,89,90,100,110,103,120,121 |
| `nvcc --list-gpu-code` | sm_ equivalents of the above | sm_ equivalents of the above |
| Host compiler used successfully | `gcc-14` / `g++-14` (Debian 14.3.0-14) | `gcc-14` / `g++-14` (same; `run_all.sh` comment confirms 13.3 requires g++-14 specifically, not gcc-14, for guard-symbol reasons — confirmed compiling with `-ccbin gcc-14` succeeded in my probe below regardless) |
| glibc 2.42 patch status | **Already applied** — confirmed `__THROW` present on `cospi` in `/usr/local/cuda-12.8/targets/x86_64-linux/include/crt/math_functions.h` | N/A (13.3 doesn't need it per README) |

**Architectures supported by BOTH toolkits** (intersection of the two
`--list-gpu-arch` outputs above): **sm_75, sm_80, sm_86, sm_87, sm_89,
sm_90** — exactly 6 targets. Notably **sm_70 is in 12.8's list but NOT in
13.3's** — CUDA 13.3 dropped Volta support. This directly affects the task's
"preferred architecture targets" list (§9 of the instructions), which
included sm_70.

Only **sm_89** has matching physical hardware (the RTX 4050 Laptop GPU).
sm_75/80/86/87/90 are compile-target evidence only — this will be stated
explicitly on every artifact record and in every claim, per the task's
requirement not to claim runtime validation for unmatched targets.

**ptxas optimization flag syntax**, confirmed from `ptxas -h` on both
versions:
```
--opt-level <N>    (-O)     Specify optimization level.   Default value: 3.
```
`-Xptxas -O0` / `-O1` / `-O2` / `-O3` is therefore the correct, verified
passthrough syntax on both installed toolkits, consistent with the task's
required primary device-optimization axis.

### 4.5 Windows-native CUDA (NOT usable for this study without extra setup)

A separate CUDA 13.3 install exists directly on Windows at
`C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin\` (`nvcc`,
`ptxas`, `cuobjdump` all present, same V13.3.33 build as the WSL2 copy).

**Attempted a trivial compile** (`nvcc -arch=sm_89 -c src/dse_minimal.cu`)
directly against this toolkit from Git Bash:
```
nvcc fatal   : Cannot find compiler 'cl.exe' in PATH
```
nvcc on Windows requires MSVC (`cl.exe`) as the host compiler; there is no
MinGW/gcc support path. `cl.exe` was not found in `PATH`, and while Visual
Studio 2019 Build Tools appear installed (`C:\Program Files (x86)\Microsoft
Visual Studio\2019\`), locating and wiring up `cl.exe`, `vcvarsall.bat`, and
the matching Windows SDK is a nontrivial environment change I have not made
and will not make without explicit approval (task instructions: no
system-wide compiler configuration changes without approval). MinGW gcc
6.3.0 is present at `C:\MinGW\bin\gcc.exe` but is **not supported** by nvcc
as a host compiler on Windows at all (nvcc's Windows host-compiler detection
only recognizes MSVC).

`docs/RESULTS.md` independently corroborates that the Windows-native 13.3
install was already found unusable for *runtime* execution
("cudaErrorInsufficientDriver, error 35") during the original artifact's
build — consistent with it never having been made to work end-to-end on
this machine.

**Conclusion: all compilation for this study must happen inside WSL2**,
using the two already-working toolkits (12.8 and 13.3) documented above. The
Windows-native toolkit is out of scope unless the user explicitly wants to
invest in setting up MSVC, which I am not doing unilaterally.

### 4.6 End-to-end toolchain validation performed

To avoid assuming the above works, I ran one throwaway probe (not committed,
lives only in WSL2 `/tmp/audit_probe`):

```
/usr/local/cuda-12.8/bin/nvcc -arch=sm_89 -O3 -Xptxas -O0 -ccbin gcc-14 -cubin -o probe128.cubin dse_minimal.cu
  → exit 0, one expected #550-D warning on pure_dead_naive's unused buf
/usr/local/cuda-13.3/bin/nvcc -arch=sm_86 -O3 -Xptxas -O3 -ccbin gcc-14 -cubin -o probe133.cubin dse_minimal.cu
  → exit 0, same expected warning
/usr/local/cuda-12.8/bin/cuobjdump -sass probe128.cubin | grep -E "Function|STS"
  → pure_dead_naive: 0 STS lines (confirms whole-allocation elimination, matches RESULTS.md)
  → pure_dead_volatile, nosync_naive, nosync_volatile: STS lines present (confirms stores survive)
```
cubin sizes: 14,560 bytes (12.8/sm_89) and 15,968 bytes (13.3/sm_86) for a
4-kernel file — gives a concrete per-artifact size baseline for disk
estimates below (§6).

This independently reproduces the documented Finding (a)/(c) behavior on
this exact machine, confirming the toolchain, GPU driver passthrough (for
later optional runtime checks), and disassembly pipeline all function
correctly before any large-scale run is attempted.

---

## 5. Conflicts and corrections vs. the task's assumed setup

1. **Platform mismatch**: task text and repo docs assume WSL2 is already the
   active shell; the actual invocation environment is native Windows Git
   Bash. Resolution: every compilation/disassembly command in this study
   will be issued via `wsl.exe -d kali-linux -- bash -c '...'`, operating on
   files under `/mnt/c/Tripti Sharma/lmhc-forge/robustness_study/` (the
   canonical, git-tracked location) rather than the untracked
   `/opt/lmhc-forge` copy.
2. **sm_70 unsupported on CUDA 13.3**: the task's preferred 6-target list
   (`sm_70, sm_75, sm_80, sm_86, sm_89, sm_90`) is not simultaneously valid
   across "all locally available selected CUDA/ptxas versions" as required,
   because 13.3 dropped sm_70. **Proposed correction**: substitute `sm_87`
   for `sm_70`, giving `sm_75, sm_80, sm_86, sm_87, sm_89, sm_90` — the
   exact intersection of both toolkits, still 6 targets, still spans
   Turing→Hopper-adjacent generations (sm_87 is Jetson Orin/Orin NX, an
   Ada-generation embedded part). `sm_70` compiles fine under 12.8 alone and
   can optionally be recorded as a 12.8-only bonus cell if desired, but is
   excluded from the core "both toolkits" factorial to keep the matrix
   rectangular. **This needs your confirmation** (see questions below).
3. **RAM ceiling inside WSL2 (7.6 GB) is much tighter than the 16 GB visible
   to Windows.** The task's worker-count formula
   (`min(4, max(1, physical_cores // 2))`) evaluates to `min(4, 5) = 4` using
   the 10 physical cores reported by Windows, but that ignores the WSL2
   memory ceiling. I will benchmark peak `ptxas`/`nvcc` RSS during the pilot
   (Phase 3) before committing to a worker count, and will cap workers below
   4 if memory pressure appears — this is exactly what the task's resource-
   control section asks for, just flagging that the naive core-count formula
   alone is insufficient here.
4. **Repo lives on the Windows-side `C:` drive (114 GB free), not the WSL2
   native filesystem (952 GB free).** All new files must go under
   `robustness_study/` inside the git repo per the task's structure
   requirement, which means compilation I/O crosses the 9p `/mnt/c` bridge.
   This is known to be slower than native ext4 for large counts of small
   file operations. Given artifact sizes are small (~15 KB cubin, SASS
   likely tens of KB per single-kernel artifact, based on the probe above),
   I don't expect this to be a hard blocker for ~1,000–2,000 artifacts, but
   I will measure actual per-artifact compile+write time in the pilot and
   report it honestly rather than assume it's fine.
5. **No conflict found** between the paper's claims and the repository's
   existing measured evidence — `docs/RESULTS.md` and
   `results/dse_differential.json` are internally consistent with each
   other and with what I independently reproduced in §4.6.

---

## 6. Disk, worker-count, and runtime estimates

- **Disk**: single-kernel probe cubins were ~15 KB; SASS text for a 4-kernel
  file via `cuobjdump` was on the order of tens of KB total (not separately
  measured per-kernel here, will be measured precisely in the pilot). Budget
  estimate: ≤500 KB per artifact including cubin + SASS + stdout/stderr logs
  + JSON metadata, generously. At 2,016 core artifacts that's ≤1 GB —
  trivial against 114 GB free on `C:`. Not a blocking constraint.
- **Workers**: start the pilot at **1 worker**, measure wall time and peak
  RSS per compile, then set the real worker count from that measurement,
  capped at `min(4, 10 physical cores // 2) = 4` per the task's formula,
  further capped by whatever the 7.6 GB WSL2 RAM ceiling actually allows.
  I will report the chosen number and why in `VALIDATION_REPORT.md`.
- **Runtime**: not yet measured at scale — the two probe compiles above
  completed in a few seconds each including nvcc/ptxas startup overhead, but
  I have not yet run the ~40-artifact pilot that would give a defensible
  per-artifact average. I will report real numbers after Phase 3, not
  estimate them here.

---

## 7. Recommended maximum parallel compilation workers (provisional)

**Provisional: 1 worker for the pilot; up to 4 for the core matrix, pending
pilot memory measurement.** Finalized in `VALIDATION_REPORT.md` after Phase 3.

---

## 8. Summary of what is and is not yet done

Done:
- Full repository inspection of every file listed in the task's §2.
- Real `git status`/branch/commit capture; branch created from clean main.
- Full environment inventory: OS, both CUDA toolkits (paths, versions,
  supported archs, ptxas flag syntax), host compiler, Python + packages,
  GPU/driver, CPU, RAM (both Windows and WSL2), disk (both filesystems).
- One real, throwaway, end-to-end compile→disassemble probe confirming the
  toolchain works and reproduces documented prior findings.
- Identified and resolved (pending your confirmation) the sm_70/CUDA 13.3
  conflict and the WSL2-vs-Windows platform mismatch.

Not yet done (by design — this is the Phase 0 gate):
- No kernel generator, compiler-matrix runner, SASS parser, classifier, or
  any experiment code has been written.
- No pilot, no core matrix, no compilation beyond the single throwaway probe
  above has been run.
- `PLAN.md`, `DATA_DICTIONARY.md`, `configs/`, `scripts/`, `kernels/`,
  `tests/`, `artifacts/`, `results/`, `figures/` under `robustness_study/`
  do not exist yet — they are proposed in my response, not created.
