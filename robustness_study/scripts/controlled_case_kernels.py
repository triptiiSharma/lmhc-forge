#!/usr/bin/env python3
"""
controlled_case_kernels.py — dedicated source templates for the 6 controlled
cases (N3, N4, N5, N6, N7, N8) that have no equivalent among the existing
Z0-Z6 core-matrix constructs. N0, N1, N2, N9, N10, N12 are REAL_COMPILED_CASE
references to existing core-matrix cells (see build_controlled_cases.py).
N11 is a REAL_BINARY_INPUT_CASE (no new source needed).

All six share the K2-shaped host context (fill -> use -> scrub, the
realistic pattern from PLAN.md, NWORDS=1024, BLOCK=32) so the only variable
under test is the scrub construct's structure, not the surrounding context.

N5 and N6 are compiled but MUST NEVER be executed (task instruction) --
N5 has intentionally invalid instruction ordering that would be meaningless
to run, N6 has a genuinely uncovered cleanup path for a subset of threads
(divergent early return before the scrub).
"""

PREAMBLE = (
    "    int tid = threadIdx.x;\n"
    "    __shared__ uint32_t sbuf[1024];\n"
    "    for (int i = tid; i < 1024; i += 32)\n"
    "        sbuf[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;\n"
    "    __syncthreads();\n"
    "    if (tid == 0) *output = sbuf[0] ^ sbuf[1023];\n"
    "    __syncthreads();\n"
)

BODIES = {
    "n3": (
        "// N3 -- zero stores AND fence present, final barrier ABSENT\n",
        PREAMBLE +
        "    volatile uint32_t *v = sbuf;\n"
        "    for (int i = tid; i < 1024; i += 32)\n"
        "        v[i] = 0u;\n"
        "    __threadfence_block();\n"
        "    // intentionally no trailing __syncthreads()\n"
    ),
    "n4": (
        "// N4 -- barrier present, scrub stores ABSENT\n",
        PREAMBLE +
        "    // intentionally no scrub store loop at all\n"
        "    __syncthreads();\n"
    ),
    "n5": (
        "// N5 -- stores, fence, barrier present but in INTENTIONALLY INVALID order.\n"
        "// DO NOT EXECUTE -- compiled/disassembled only.\n",
        PREAMBLE +
        "    __syncthreads();\n"
        "    __threadfence_block();\n"
        "    volatile uint32_t *v = sbuf;\n"
        "    for (int i = tid; i < 1024; i += 32)\n"
        "        v[i] = 0u;\n"
    ),
    "n6": (
        "// N6 -- multiple source-level exits; the cleanup path for tid<16 is\n"
        "// INTENTIONALLY UNCOVERED (early return before the scrub).\n"
        "// DO NOT EXECUTE -- divergent early return, compiled/disassembled only.\n",
        PREAMBLE +
        "    if (tid < 16) return;   // early exit: these threads never reach the scrub below\n"
        "    volatile uint32_t *v = sbuf;\n"
        "    for (int i = tid; i < 1024; i += 32)\n"
        "        v[i] = 0u;\n"
        "    __threadfence_block();\n"
        "    __syncthreads();\n"
    ),
    "n7": (
        "// N7 -- intended scrub stores write a NONZERO value (0xAA), not zero.\n",
        PREAMBLE +
        "    volatile uint32_t *v = sbuf;\n"
        "    for (int i = tid; i < 1024; i += 32)\n"
        "        v[i] = 0xAAu;\n"
        "    __threadfence_block();\n"
        "    __syncthreads();\n"
    ),
    "n8": (
        "// N8 -- full recognized signature on sbuf, PLUS an unrelated,\n"
        "// unscrubbed shared-memory buffer (junk).\n",
        "    int tid = threadIdx.x;\n"
        "    __shared__ uint32_t sbuf[1024];\n"
        "    __shared__ uint32_t junk[32];\n"
        "    for (int i = tid; i < 1024; i += 32)\n"
        "        sbuf[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;\n"
        "    __syncthreads();\n"
        "    if (tid == 0) *output = sbuf[0] ^ sbuf[1023];\n"
        "    junk[tid] = tid;   // unrelated, never scrubbed\n"
        "    __syncthreads();\n"
        "    volatile uint32_t *v = sbuf;\n"
        "    for (int i = tid; i < 1024; i += 32)\n"
        "        v[i] = 0u;\n"
        "    __threadfence_block();\n"
        "    __syncthreads();\n"
    ),
}

# Case metadata: intended structural condition, whether safe to execute,
# and category tag.
CASE_META = {
    "n3": {"id": "N3", "intended_category": "PARTIAL_SIGNATURE",
           "description": "zero stores and fence present, final barrier absent",
           "safe_to_execute": True},
    "n4": {"id": "N4", "intended_category": "ABSENT",
           "description": "barrier present, scrub stores absent",
           "safe_to_execute": True},
    "n5": {"id": "N5", "intended_category": "PARTIAL_SIGNATURE",
           "description": "stores, fence, barrier present in invalid order",
           "safe_to_execute": False},
    "n6": {"id": "N6", "intended_category": "PARTIAL_SIGNATURE_OR_UNDECIDABLE",
           "description": "multi-exit, one cleanup path intentionally uncovered",
           "safe_to_execute": False},
    "n7": {"id": "N7", "intended_category": "ABSENT",
           "description": "intended scrub stores write a nonzero value (0xAA) -- the checker "
                           "must not be fooled by fence+barrier presence into treating a "
                           "nonzero store as a zero-store; correct behavior is ABSENT "
                           "despite fence/barrier being present",
           "safe_to_execute": True},
    "n8": {"id": "N8", "intended_category": "FULL_SIGNATURE",
           "description": "full signature plus unrelated shared-memory stores",
           "safe_to_execute": True},
}


def render(case_key: str) -> tuple[str, str]:
    comment, body = BODIES[case_key]
    fname = f"kernel_{case_key}"
    src = (
        "#include <stdint.h>\n"
        "#include <cuda_runtime.h>\n\n"
        f"{comment}"
        f"__launch_bounds__(32, 1)\n"
        f"__global__ void {fname}(uint32_t *output)\n"
        "{\n"
        f"{body}"
        "}\n\n"
        "int main() {\n"
        "    uint32_t *d_out;\n"
        "    cudaMalloc(&d_out, sizeof(uint32_t));\n"
        f"    {fname}<<<1, 32>>>(d_out);\n"
        "    cudaDeviceSynchronize();\n"
        "    cudaFree(d_out);\n"
        "    return 0;\n"
        "}\n"
    )
    return fname, src
