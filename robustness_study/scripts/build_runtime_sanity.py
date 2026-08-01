#!/usr/bin/env python3
"""
build_runtime_sanity.py — Section H: bounded runtime sanity subset.

6 constructs (Z0,Z1,Z2,Z4,Z5,Z6) x 3 contexts (K2,K3,K4) = 18 diagnostic
variants, sm_89 / CUDA 12.8 only (the only real hardware in this study).
K0 is explicitly excluded (instrumented readback would make the pure-dead
allocation "live," invalidating the property K0 is designed to study).

Each diagnostic variant is the SAME construct x context kernel body from
kernel_lib.py, with one addition: after the logical cleanup point (after
the Z-construct's scrub code, before any context postamble), every thread
copies its own words of the sensitive buffer to a global-memory output
array, so the host can read back and count nonzero words post-scrub.

This is explicitly NOT a new Exposure Index / residue-recovery experiment
(no cross-kernel co-residency, no writer/listener pattern) -- it is a
same-kernel, same-launch behavioral sanity check: did this specific
compiled template launch, synchronize, compute its ordinary result
correctly, and leave the expected zero/nonzero pattern in the buffer
immediately after its own cleanup point.

Barrier-unsafe or genuinely divergent-early-return constructs are never
included here (the 6 constructs used are all from the validated, safe
Z0-Z6 set; N5/N6 controlled cases are never run).
"""

import json
import subprocess
import sys
import time
import datetime
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from kernel_lib import CONSTRUCTS, CONTEXTS, Construct, Context, SBUF

CONSTRUCT_IDS = ["Z0", "Z1", "Z2", "Z4", "Z5", "Z6"]
CONTEXT_IDS = ["K2", "K3", "K4"]
NVCC = "/usr/local/cuda-12.8/bin/nvcc"
ARCH = "sm_89"

OUT_DIR = ROOT / "kernels" / "runtime_sanity"
BIN_DIR = ROOT / "artifacts" / "runtime_sanity"
N_REPEATS = 5


def render_diagnostic(construct: Construct, context: Context) -> tuple[str, str, dict]:
    """Same body as kernel_lib.render_kernel, plus a post-cleanup readback
    of SBUF into a global `diag_out` array."""
    fname = f"diag_{construct.id.lower()}_{context.id.lower()}"
    scrub = construct.scrub_code(SBUF, context.nwords, context.block, context.elem)
    readback = (
        f"    for (int i = tid; i < {context.nwords}; i += {context.block})\n"
        f"        diag_out[i] = (uint32_t){SBUF}[i];\n"
    ) if context.elem == "uint32_t" else (
        f"    for (int i = tid; i < {context.nwords}; i += {context.block})\n"
        f"        diag_out[i] = __float_as_uint({SBUF}[i]);\n"
    )
    params = context.params + f", uint32_t *diag_out"
    body = context.preamble + scrub + readback + context.postamble
    src = (
        f"__launch_bounds__({context.block}, 1)\n"
        f"__global__ void {fname}({params})\n"
        "{\n"
        f"{body}"
        "}\n"
    )
    meta = {"fname": fname, "nwords": context.nwords, "block": context.block, "elem": context.elem}
    return fname, src, meta


def make_host_main(fname: str, context: Context, meta: dict) -> str:
    nwords = meta["nwords"]
    if context.id == "K2":
        return (
            "int main() {\n"
            "    uint32_t *d_out, *d_diag;\n"
            "    cudaMalloc(&d_out, sizeof(uint32_t));\n"
            f"    cudaMalloc(&d_diag, {nwords} * sizeof(uint32_t));\n"
            f"    for (int rep = 0; rep < {N_REPEATS}; rep++) {{\n"
            f"        {fname}<<<1, {context.block}>>>(d_out, d_diag);\n"
            "        cudaError_t launch_err = cudaGetLastError();\n"
            "        cudaError_t sync_err = cudaDeviceSynchronize();\n"
            "        uint32_t h_out;\n"
            f"        uint32_t h_diag[{nwords}];\n"
            "        cudaMemcpy(&h_out, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);\n"
            f"        cudaMemcpy(h_diag, d_diag, {nwords} * sizeof(uint32_t), cudaMemcpyDeviceToHost);\n"
            "        int nonzero = 0;\n"
            f"        for (int i = 0; i < {nwords}; i++) if (h_diag[i] != 0u) nonzero++;\n"
            "        printf(\"REP %d launch=%s sync=%s output=%u nonzero_words=%d\\n\", rep,\n"
            "               cudaGetErrorString(launch_err), cudaGetErrorString(sync_err), h_out, nonzero);\n"
            "    }\n"
            "    cudaFree(d_out); cudaFree(d_diag);\n"
            "    return 0;\n"
            "}\n"
        )
    if context.id == "K3":
        return (
            "int main() {\n"
            "    float *d_in, *d_out; uint32_t *d_diag;\n"
            "    float h_in[256];\n"
            "    for (int i = 0; i < 256; i++) h_in[i] = (float)(i + 1);\n"
            "    cudaMalloc(&d_in, 256 * sizeof(float));\n"
            "    cudaMalloc(&d_out, sizeof(float));\n"
            f"    cudaMalloc(&d_diag, {nwords} * sizeof(uint32_t));\n"
            "    cudaMemcpy(d_in, h_in, 256 * sizeof(float), cudaMemcpyHostToDevice);\n"
            f"    for (int rep = 0; rep < {N_REPEATS}; rep++) {{\n"
            f"        {fname}<<<1, {context.block}>>>(d_in, d_out, 256, d_diag);\n"
            "        cudaError_t launch_err = cudaGetLastError();\n"
            "        cudaError_t sync_err = cudaDeviceSynchronize();\n"
            "        float h_out;\n"
            f"        uint32_t h_diag[{nwords}];\n"
            "        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);\n"
            f"        cudaMemcpy(h_diag, d_diag, {nwords} * sizeof(uint32_t), cudaMemcpyDeviceToHost);\n"
            "        float expected = 0.0f;\n"
            "        for (int i = 0; i < 256; i++) expected += h_in[i];\n"
            "        int nonzero = 0;\n"
            f"        for (int i = 0; i < {nwords}; i++) if (h_diag[i] != 0u) nonzero++;\n"
            "        printf(\"REP %d launch=%s sync=%s output=%f expected=%f match=%d nonzero_words=%d\\n\", rep,\n"
            "               cudaGetErrorString(launch_err), cudaGetErrorString(sync_err), h_out, expected,\n"
            "               (h_out == expected), nonzero);\n"
            "    }\n"
            "    cudaFree(d_in); cudaFree(d_out); cudaFree(d_diag);\n"
            "    return 0;\n"
            "}\n"
        )
    if context.id == "K4":
        return (
            "int main() {\n"
            "    float *dA, *dB, *dC; uint32_t *d_diag;\n"
            "    float hA[256], hB[256];\n"
            "    for (int i = 0; i < 256; i++) { hA[i] = (float)(i % 7); hB[i] = (float)(i % 5); }\n"
            "    cudaMalloc(&dA, 256 * sizeof(float));\n"
            "    cudaMalloc(&dB, 256 * sizeof(float));\n"
            "    cudaMalloc(&dC, 256 * sizeof(float));\n"
            f"    cudaMalloc(&d_diag, {nwords} * sizeof(uint32_t));\n"
            "    cudaMemcpy(dA, hA, 256 * sizeof(float), cudaMemcpyHostToDevice);\n"
            "    cudaMemcpy(dB, hB, 256 * sizeof(float), cudaMemcpyHostToDevice);\n"
            f"    for (int rep = 0; rep < {N_REPEATS}; rep++) {{\n"
            f"        {fname}<<<1, {context.block}>>>(dA, dB, dC, 16, 16, 16, d_diag);\n"
            "        cudaError_t launch_err = cudaGetLastError();\n"
            "        cudaError_t sync_err = cudaDeviceSynchronize();\n"
            "        float hC[256];\n"
            f"        uint32_t h_diag[{nwords}];\n"
            "        cudaMemcpy(hC, dC, 256 * sizeof(float), cudaMemcpyDeviceToHost);\n"
            f"        cudaMemcpy(h_diag, d_diag, {nwords} * sizeof(uint32_t), cudaMemcpyDeviceToHost);\n"
            "        float expected[256];\n"
            "        int ok = 1;\n"
            "        for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) {\n"
            "            float acc = 0.0f;\n"
            "            for (int k = 0; k < 16; k++) acc += hA[r*16+k] * hB[k*16+c];\n"
            "            expected[r*16+c] = acc;\n"
            "            if (hC[r*16+c] != acc) ok = 0;\n"
            "        }\n"
            "        int nonzero = 0;\n"
            f"        for (int i = 0; i < {nwords}; i++) if (h_diag[i] != 0u) nonzero++;\n"
            "        printf(\"REP %d launch=%s sync=%s gemm_correct=%d nonzero_words=%d\\n\", rep,\n"
            "               cudaGetErrorString(launch_err), cudaGetErrorString(sync_err), ok, nonzero);\n"
            "    }\n"
            "    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(d_diag);\n"
            "    return 0;\n"
            "}\n"
        )
    raise ValueError(context.id)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BIN_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    for zid in CONSTRUCT_IDS:
        for kid in CONTEXT_IDS:
            construct = CONSTRUCTS[zid]
            context = CONTEXTS[kid]
            fname, kernel_src, meta = render_diagnostic(construct, context)
            main_src = make_host_main(fname, context, meta)
            full_src = "#include <stdint.h>\n#include <stdio.h>\n#include <cuda_runtime.h>\n\n" + kernel_src + "\n" + main_src

            src_path = OUT_DIR / f"{fname}.cu"
            src_path.write_text(full_src, encoding="utf-8", newline="\n")

            exe_path = BIN_DIR / fname
            cmd = [NVCC, f"-arch={ARCH}", "-O3", "-Xptxas", "-O3", "-ccbin", "gcc-14",
                   "-o", str(exe_path), str(src_path)]
            t0 = time.time()
            cr = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
            compile_ms = int((time.time() - t0) * 1000)

            record = {
                "zeroization_construct": zid, "kernel_context": kid,
                "diagnostic_kernel_name": fname,
                "source_file": str(src_path.as_posix()),
                "compile_command": " ".join(cmd),
                "compile_return_code": cr.returncode,
                "compile_time_ms": compile_ms,
                "arch": ARCH, "cuda_version": "cuda-12.8",
                "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            }
            if cr.returncode != 0:
                record["compile_status"] = "COMPILE_FAILED"
                record["error_message"] = cr.stderr.strip()[:1000]
                results.append(record)
                print(f"  {zid}x{kid}: COMPILE_FAILED: {cr.stderr.strip()[:200]}")
                continue
            record["compile_status"] = "COMPILE_OK"
            if cr.stderr.strip():
                record["compile_warnings"] = cr.stderr.strip()[:500]

            rr = subprocess.run([str(exe_path)], capture_output=True, text=True, timeout=30)
            record["run_return_code"] = rr.returncode
            record["run_stdout"] = rr.stdout
            record["run_stderr"] = rr.stderr.strip()[:1000] if rr.stderr else None
            results.append(record)
            print(f"  {zid}x{kid}: run_rc={rr.returncode}")
            for line in rr.stdout.strip().splitlines():
                print(f"    {line}")

    out_path = ROOT / "results" / "runtime_sanity.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {out_path} ({len(results)} records)")


if __name__ == "__main__":
    main()
