#!/usr/bin/env python3
"""
kernel_lib.py — single source of truth for the Z0-Z6 zeroization constructs
and K0-K5 kernel contexts defined in PLAN.md §1-2.

Every other script (generate_kernels.py, tests) imports CONSTRUCTS and
CONTEXTS from here rather than re-encoding the templates, so the design in
PLAN.md and the generated corpus cannot silently drift apart.

Design invariants (verified by dev_notes_phase1/scratch_feasibility_probe.cu
and scratch_runtime_check.cu during Phase 1 — see PLAN.md §9):
  - Every context's NWORDS is divisible by its BLOCK, so the scrub loop has
    a uniform per-thread trip count and never diverges.
  - __syncthreads()/__threadfence_block() in every construct are called
    unconditionally by all threads — never inside a tid-dependent branch.
  - The scrub injection point sits after the context's last "live" use of
    SBUF and before any divergent tail code (K4's output write).
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Construct:
    id: str
    name: str
    role: str            # "negative_control" | "primary_variable" | "partial_case" | "intended_full_signature"
    expected_structural_requirement: str

    def scrub_code(self, sbuf: str, nwords: int, block: int, elem: str) -> str:
        zero = "0u" if elem == "uint32_t" else "0.0f"
        if self.id == "Z0":
            return ""
        if self.id == "Z1":
            return (
                f"    for (int i = tid; i < {nwords}; i += {block})\n"
                f"        {sbuf}[i] = {zero};\n"
            )
        if self.id == "Z2":
            return (
                f"    volatile {elem} *v_{sbuf} = {sbuf};\n"
                f"    for (int i = tid; i < {nwords}; i += {block})\n"
                f"        v_{sbuf}[i] = {zero};\n"
            )
        if self.id == "Z3":
            return (
                f"    for (int i = tid; i < {nwords}; i += {block})\n"
                f"        {sbuf}[i] = {zero};\n"
                f"    __syncthreads();\n"
            )
        if self.id == "Z4":
            return (
                f"    volatile {elem} *v_{sbuf} = {sbuf};\n"
                f"    for (int i = tid; i < {nwords}; i += {block})\n"
                f"        v_{sbuf}[i] = {zero};\n"
                f"    __syncthreads();\n"
            )
        if self.id == "Z5":
            return (
                f"    volatile {elem} *v_{sbuf} = {sbuf};\n"
                f"    for (int i = tid; i < {nwords}; i += {block})\n"
                f"        v_{sbuf}[i] = {zero};\n"
                f"    __threadfence_block();\n"
                f"    __syncthreads();\n"
            )
        if self.id == "Z6":
            return (
                f"    for (int i = tid; i < {nwords}; i += {block}) {{\n"
                f"        unsigned int saddr_{sbuf} = __cvta_generic_to_shared(&{sbuf}[i]);\n"
                f'        asm volatile("st.shared.u32 [%0], %1;" : : "r"(saddr_{sbuf}), "r"(0u) : "memory");\n'
                f"    }}\n"
                f"    __threadfence_block();\n"
                f"    __syncthreads();\n"
            )
        raise ValueError(f"unknown construct {self.id}")


CONSTRUCTS: dict[str, Construct] = {
    c.id: c for c in [
        Construct("Z0", "no_scrub", "negative_control",
                   "none — ABSENT is the correct classification"),
        Construct("Z1", "plain_zeroing_loop", "primary_variable",
                   "zero-stores present or eliminated (compiler-dependent; this is what's measured)"),
        Construct("Z2", "volatile_zeroing_loop", "partial_case",
                   "zero-stores present; no fence/barrier ordering signature"),
        Construct("Z3", "plain_loop_plus_barrier", "partial_case",
                   "zero-stores present-or-eliminated + BAR.SYNC; no MEMBAR.SC.CTA"),
        Construct("Z4", "volatile_loop_plus_barrier", "partial_case",
                   "zero-stores present + BAR.SYNC; no MEMBAR.SC.CTA"),
        Construct("Z5", "hardened_lmhc", "intended_full_signature",
                   "zero-stores + MEMBAR.SC.CTA + BAR.SYNC, in order, before EXIT"),
        Construct("Z6", "inline_ptx_hardened", "intended_full_signature",
                   "zero-stores + MEMBAR.SC.CTA + BAR.SYNC, in order, before EXIT"),
    ]
}


@dataclass(frozen=True)
class Context:
    id: str
    name: str
    block: int
    nwords: int
    elem: str             # "uint32_t" | "float"
    based_on: str
    params: str            # extra kernel parameter list (after the common ones are added by caller)
    preamble: str           # body before the scrub injection point; must reference sbuf/tid via .format()
    postamble: str          # body after the scrub injection point (usually empty)

    def __post_init__(self):
        if self.nwords % self.block != 0:
            raise ValueError(
                f"{self.id}: NWORDS={self.nwords} not divisible by BLOCK={self.block} "
                f"— would produce a non-uniform per-thread trip count"
            )


SBUF = "sbuf"  # uniform sensitive-buffer variable name used by every context/construct

CONTEXTS: dict[str, Context] = {
    c.id: c for c in [
        Context(
            id="K0", name="pure_dead_allocation", block=32, nwords=1024, elem="uint32_t",
            based_on="dse_minimal.cu: pure_dead_naive / pure_dead_volatile",
            params="uint32_t *output",
            preamble=(
                "    int tid = threadIdx.x;\n"
                f"    __shared__ uint32_t {SBUF}[1024];\n"
                "    if (tid == 0) *output = 0u;\n"
            ),
            postamble="",
        ),
        Context(
            id="K1", name="fill_then_scrub", block=32, nwords=1024, elem="uint32_t",
            based_on="dse_minimal.cu fill phase (nosync_*), output made independent of buf per task's K1 definition",
            params="uint32_t *output",
            preamble=(
                "    int tid = threadIdx.x;\n"
                f"    __shared__ uint32_t {SBUF}[1024];\n"
                "    for (int i = tid; i < 1024; i += 32)\n"
                f"        {SBUF}[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;\n"
                "    if (tid == 0) *output = (uint32_t)tid;\n"
            ),
            postamble="",
        ),
        Context(
            id="K2", name="fill_use_then_scrub", block=32, nwords=1024, elem="uint32_t",
            based_on="dse_test.cu: scrub_naive / scrub_hardened (exact shape)",
            params="uint32_t *output",
            preamble=(
                "    int tid = threadIdx.x;\n"
                f"    __shared__ uint32_t {SBUF}[1024];\n"
                "    for (int i = tid; i < 1024; i += 32)\n"
                f"        {SBUF}[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;\n"
                "    __syncthreads();\n"
                f"    if (tid == 0) *output = {SBUF}[0] ^ {SBUF}[1023];\n"
                "    __syncthreads();\n"
            ),
            postamble="",
        ),
        Context(
            id="K3", name="reduction_like", block=256, nwords=256, elem="float",
            based_on="test_kernels.cu: reduction_sum; empirically verified correct on RTX 4050 for Z2 (PLAN.md §9)",
            params="const float *input, float *output, int n",
            preamble=(
                "    int tid = threadIdx.x;\n"
                f"    __shared__ float {SBUF}[256];\n"
                f"    {SBUF}[tid] = (tid < n) ? input[blockIdx.x * 256 + tid] : 0.0f;\n"
                "    __syncthreads();\n"
                "    for (int stride = 128; stride > 0; stride >>= 1) {\n"
                f"        if (tid < stride) {SBUF}[tid] += {SBUF}[tid + stride];\n"
                "        __syncthreads();\n"
                "    }\n"
                f"    if (tid == 0) *output = {SBUF}[0];\n"
            ),
            postamble="",
        ),
        Context(
            id="K4", name="tiled_gemm_like", block=256, nwords=256, elem="float",
            based_on="test_kernels.cu: gemm_tile; a_tile (renamed sbuf) is the confidential/sensitive one",
            params="const float *A, const float *B, float *C, int M, int N, int Kdim",
            preamble=(
                "    int tid = threadIdx.x;\n"
                "    int tx = tid % 16;\n"
                "    int ty = tid / 16;\n"
                "    int row = blockIdx.y * 16 + ty;\n"
                "    int col = blockIdx.x * 16 + tx;\n"
                f"    __shared__ float {SBUF}[256];\n"
                "    __shared__ float b_tile[256];\n"
                "    float acc = 0.0f;\n"
                "    for (int t = 0; t < (Kdim + 15) / 16; t++) {\n"
                "        int a_col = t * 16 + tx;\n"
                "        int b_row = t * 16 + ty;\n"
                f"        {SBUF}[ty * 16 + tx] = (row < M && a_col < Kdim) ? A[row * Kdim + a_col] : 0.0f;\n"
                "        b_tile[ty * 16 + tx] = (b_row < Kdim && col < N) ? B[b_row * N + col] : 0.0f;\n"
                "        __syncthreads();\n"
                "        for (int k = 0; k < 16; k++)\n"
                f"            acc += {SBUF}[ty * 16 + k] * b_tile[k * 16 + tx];\n"
                "        __syncthreads();\n"
                "    }\n"
            ),
            postamble=(
                "    if (row < M && col < N)\n"
                "        C[row * N + col] = acc;\n"
            ),
        ),
        Context(
            id="K5", name="attention_style", block=64, nwords=2048, elem="float",
            based_on="test_kernels.cu: attention_fwd; smem_qk (renamed sbuf) is the confidential/sensitive one",
            params="const float *q_proj, const float *k_proj, float *attn_out",
            preamble=(
                "    int tid = threadIdx.x;\n"
                "    int bid = blockIdx.x;\n"
                f"    __shared__ float {SBUF}[2048];\n"
                "    __shared__ float smem_softmax[64];\n"
                "    if (tid < 32)\n"
                f"        {SBUF}[bid * 32 + tid] = q_proj[bid * 32 + tid];\n"
                "    __syncthreads();\n"
                "    if (tid == 0) {\n"
                "        float row_sum = 0.0f;\n"
                "        for (int j = 0; j < 32; j++)\n"
                f"            row_sum += {SBUF}[j] * k_proj[j];\n"
                "        smem_softmax[bid] = row_sum;\n"
                "    }\n"
                "    __syncthreads();\n"
                "    if (tid == 0) attn_out[bid] = smem_softmax[bid];\n"
                "    __syncthreads();\n"
            ),
            postamble="",
        ),
    ]
}


def render_kernel(construct: Construct, context: Context) -> tuple[str, str]:
    """Return (function_name, full __global__ kernel source) for one (Z,K) pair."""
    fname = f"kernel_{construct.id.lower()}_{context.id.lower()}"
    scrub = construct.scrub_code(SBUF, context.nwords, context.block, context.elem)
    body = context.preamble + scrub + context.postamble
    src = (
        f"__launch_bounds__({context.block}, 1)\n"
        f"__global__ void {fname}({context.params})\n"
        "{\n"
        f"{body}"
        "}\n"
    )
    return fname, src
