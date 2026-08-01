#!/usr/bin/env python3
"""verifier_perf.py -- RP Sec VI.F / Table VI: independent-verifier cost.

Wraps attester.py (unmodified -- this script does not alter Module 5's
verification logic) to measure, per binary in the test corpus:
  - wall-clock time for the full attester.py run (subprocess, time.perf_counter)
  - cubin_size_bytes (surfaced from witness.json, already computed by Module 4)
  - CFG-reconstruction time in isolation (the cuobjdump -sass call CHECK 2
    depends on, timed separately from the rest of attester.py's work)
  - peak resident memory of the attester.py subprocess
    (resource.getrusage(RUSAGE_CHILDREN).ru_maxrss after it exits)

Also runs a SYNTHETIC scalability test: N near-duplicate toy kernels
(N=1,2,4,8), each with one __confidential__-tainted __shared__ buffer in the
same shape as gemm_tile in test_kernels.cu, run through the REAL pipeline
(discover -> rewrite -> nvcc --cubin -> witness_gen -> attester) so timing
reflects genuine tool behavior, not a hand-waved estimate. Labeled explicitly
as synthetic -- this is not a claim about real-world kernel-count
distributions.

Output: JSON to stdout -> results/verifier_perf.json
"""
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
CCBIN = "/usr/bin/gcc-14"
CUOBJDUMP = "/usr/local/cuda-12.8/bin/cuobjdump"
ARCH = "sm_89"


def time_and_rss(cmd, cwd=None):
    """Run cmd as a subprocess; return (wall_seconds, peak_rss_kb, returncode, stdout, stderr)."""
    rss_before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    t0 = time.perf_counter()
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    t1 = time.perf_counter()
    rss_after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    # ru_maxrss is cumulative peak across all reaped children in this process;
    # the delta isolates (approximately) this call's contribution when run
    # sequentially with no concurrent children, which is the case here.
    peak_rss_kb = max(rss_after - rss_before, 0) or rss_after
    return t1 - t0, peak_rss_kb, r.returncode, r.stdout, r.stderr


def measure_existing_corpus(tmpdir: Path):
    """Time attester.py on the existing test_kernels.cubin/witness.json (Module 4/5 corpus).

    NOTE: the on-disk results/witness.json predates the T5 CHECK-1 fix
    documented in docs/RESULTS.md (Module 5) -- it has no source_file_hash field,
    so it is EXPECTED to FAIL against the current (post-fix) attester.py.
    This is not a defect; docs/RESULTS.md's own T5/T6 analysis documents exactly
    this old-format-vs-new-verifier mismatch. We report that stale-witness
    FAIL timing as one data point (it still costs real wall-clock/RSS to
    discover the mismatch), and separately regenerate a FRESH, current-format
    witness (via witness_gen.py, written only to a temp dir -- results/witness.json
    on disk is never overwritten, since docs/RESULTS.md cites its exact kernel_hash
    value verbatim) to get the PASS-path cost Table VI actually wants.
    """
    cubin = REPO / "results" / "test_kernels.cubin"
    witness = REPO / "results" / "witness.json"
    if not cubin.exists() or not witness.exists():
        return {"error": f"corpus files missing: {cubin} / {witness}"}

    cubin_size = cubin.stat().st_size

    # (a) stale on-disk witness -- expected FAIL (old format, see note above)
    stale_wall_s, stale_rss_kb, stale_rc, _, _ = time_and_rss(
        [sys.executable, "src/attester.py", str(cubin), str(witness)],
        cwd=str(REPO),
    )

    # (b) fresh, current-format witness for the same source -- expected PASS
    fresh_witness = tmpdir / "test_kernels_fresh_witness.json"
    fresh_cubin = tmpdir / "test_kernels_fresh.cubin"
    gen = subprocess.run(
        [sys.executable, "src/witness_gen.py", "src/test_kernels.cu",
         "--arch", ARCH, "--nvcc", NVCC, "--ccbin", CCBIN,
         "--output", str(fresh_witness), "--save-cubin", str(fresh_cubin)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    if gen.returncode != 0 or not fresh_witness.exists():
        fresh = {"error": f"witness_gen.py (fresh) failed: {gen.stderr[-500:]}"}
    else:
        fresh_wall_s, fresh_rss_kb, fresh_rc, _, _ = time_and_rss(
            [sys.executable, "src/attester.py", str(fresh_cubin), str(fresh_witness)],
            cwd=str(REPO),
        )
        fresh = {
            "cubin_size_bytes": fresh_cubin.stat().st_size,
            "verify_wallclock_s": round(fresh_wall_s, 4),
            "verify_peak_rss_kb": fresh_rss_kb,
            "attester_exit_code": fresh_rc,
            "attester_verdict": "PASS" if fresh_rc == 0 else "FAIL",
        }

    # CFG-reconstruction time in isolation: just the cuobjdump -sass call
    cfg_wall_s, cfg_rss_kb, cfg_rc, _, cfg_err = time_and_rss(
        [CUOBJDUMP, "-sass", str(cubin)]
    )

    return {
        "binary": "test_kernels (Module 4/5 corpus)",
        "cubin_size_bytes": cubin_size,
        "stale_on_disk_witness": {
            "note": "results/witness.json predates the T5 CHECK-1 fix (no source_file_hash) -- FAIL is expected and correct, not a bug. See docs/RESULTS.md Module 5 T5/T6.",
            "verify_wallclock_s": round(stale_wall_s, 4),
            "verify_peak_rss_kb": stale_rss_kb,
            "attester_exit_code": stale_rc,
            "attester_verdict": "PASS" if stale_rc == 0 else "FAIL",
        },
        "fresh_current_format_witness": fresh,
        "cfg_reconstruct_wallclock_s": round(cfg_wall_s, 4),
        "cfg_reconstruct_peak_rss_kb": cfg_rss_kb,
    }


SYNTH_KERNEL_TEMPLATE = """
__global__ void synth_kernel_{i}(
    __confidential__ const float *A,
    const float *B,
    float *C,
    int M, int N, int K)
{{
    __shared__ float a_tile_{i}[TILE * TILE];
    __shared__ float b_tile_{i}[TILE * TILE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * TILE + ty;
    int col = blockIdx.x * TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < (K + TILE - 1) / TILE; t++) {{
        int a_col = t * TILE + tx;
        int b_row = t * TILE + ty;
        a_tile_{i}[ty * TILE + tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        b_tile_{i}[ty * TILE + tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < TILE; k++)
            acc += a_tile_{i}[ty * TILE + k] * b_tile_{i}[k * TILE + tx];
        __syncthreads();
    }}
    if (row < M && col < N) C[row * N + col] = acc;
}}
"""

SYNTH_HEADER = """// SYNTHETIC scalability test corpus -- generated by verifier_perf.py.
// N near-duplicate toy kernels, each shaped like gemm_tile (test_kernels.cu),
// for stress-testing verifier scaling with kernel count. Not a real workload.
#include <stdint.h>
#include <cuda_runtime.h>
#define __confidential__ /* confidential-param */
#define TILE 16
"""


def build_synthetic(n, tmpdir: Path):
    src = SYNTH_HEADER + "\n".join(
        SYNTH_KERNEL_TEMPLATE.format(i=i) for i in range(n)
    )
    src_path = tmpdir / f"synth_{n}.cu"
    src_path.write_text(src)
    return src_path


def measure_synthetic(n, tmpdir: Path):
    src_path = build_synthetic(n, tmpdir)
    witness_path = tmpdir / f"synth_{n}_witness.json"
    cubin_path = tmpdir / f"synth_{n}.cubin"

    t0 = time.perf_counter()
    r = subprocess.run(
        [sys.executable, "src/witness_gen.py", str(src_path),
         "--arch", ARCH, "--nvcc", NVCC, "--ccbin", CCBIN,
         "--output", str(witness_path), "--save-cubin", str(cubin_path)],
        cwd=str(REPO), capture_output=True, text=True,
    )
    gen_wall_s = time.perf_counter() - t0
    if r.returncode != 0:
        return {"n_kernels": n, "error": f"witness_gen.py failed: {r.stderr[-800:]}"}

    wall_s, rss_kb, rc, out, err = time_and_rss(
        [sys.executable, "src/attester.py", str(cubin_path), str(witness_path)],
        cwd=str(REPO),
    )
    cubin_size = cubin_path.stat().st_size if cubin_path.exists() else None

    return {
        "n_kernels": n,
        "witness_gen_wallclock_s": round(gen_wall_s, 4),
        "cubin_size_bytes": cubin_size,
        "verify_wallclock_s": round(wall_s, 4),
        "verify_peak_rss_kb": rss_kb,
        "attester_exit_code": rc,
        "attester_verdict": "PASS" if rc == 0 else "FAIL",
    }


def main():
    import tempfile
    with tempfile.TemporaryDirectory() as td0:
        existing = measure_existing_corpus(Path(td0))

    synth_results = []
    with tempfile.TemporaryDirectory() as td:
        tmpdir = Path(td)
        for n in (1, 2, 4, 8):
            print(f"[verifier_perf] synthetic scalability: N={n} kernels ...", file=sys.stderr)
            res = measure_synthetic(n, tmpdir)
            synth_results.append(res)
            print(f"  {res}", file=sys.stderr)

    out = {
        "experiment": "Independent-verifier cost (RP Sec VI.F / Table VI)",
        "existing_corpus": existing,
        "synthetic_scalability_test": {
            "label": "SYNTHETIC -- N near-duplicate toy kernels (gemm_tile-shaped), "
                     "not a claim about real-world kernel-count distributions. "
                     "Generated fresh for this measurement, run through the real "
                     "discover/rewrite/compile/witness_gen/attester pipeline.",
            "results": synth_results,
        },
        "note": "Per Phase 1.6 instructions: 'no scaling curve is fit to so few "
                "points' -- N=1,2,4,8 are reported as measured points only.",
    }
    out_path = REPO / "results" / "verifier_perf.json"
    out_path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
