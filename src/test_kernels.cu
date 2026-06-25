// test_kernels.cu — demonstration input for Modules 1 (discover) and 2 (rewriter)
//
// Contains three kernels:
//   attention_fwd  — large __shared__ buffer, __confidential__ input ptr
//   reduction_sum  — __shared__ buffer, non-sensitive input
//   gemm_tile      — two __shared__ buffers, one __confidential__ input
//
// Running discover.py --taint on this file should flag only the buffers
// that receive tainted (confidential) data; the conservative pass flags all.
// Running lmhc_rewriter.py should insert scrubs before each kernel's closing }.

#include <stdint.h>
#include <cuda_runtime.h>

// Annotation macro — maps to GCC annotate attribute for libclang-based tools.
// In device code it's ignored; Module 1 searches for this string pattern.
#define __confidential__ /* confidential-param */

#define ATTN_SEQ   64
#define ATTN_HEAD  32
#define RED_N     256
#define GEMM_TILE  16

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: attention_fwd
// q_proj is confidential (e.g., query projection from private user embedding).
// smem_qk will be tainted because it is written from q_proj.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void attention_fwd(
    __confidential__ const float *q_proj,   // confidential input
    const float                 *k_proj,    // non-confidential
    float                       *attn_out,
    int                          seq_len)
{
    __shared__ float smem_qk[ATTN_SEQ * ATTN_HEAD];   // tainted (receives q_proj)
    __shared__ float smem_softmax[ATTN_SEQ];            // tainted (derived from smem_qk)

    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Load q into smem_qk — confidential data enters shared memory
    if (tid < ATTN_HEAD)
        smem_qk[bid * ATTN_HEAD + tid] = q_proj[bid * ATTN_HEAD + tid];
    __syncthreads();

    // Compute QK^T and softmax (simplified)
    if (tid == 0) {
        float row_sum = 0.0f;
        for (int j = 0; j < ATTN_HEAD; j++)
            row_sum += smem_qk[j] * k_proj[j];
        smem_softmax[bid] = row_sum;
    }
    __syncthreads();

    if (tid == 0)
        attn_out[bid] = smem_softmax[bid];
    __syncthreads();

    // single exit — lmhc_rewriter will insert scrub here
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: reduction_sum
// Non-confidential: conservative pass flags smem_partial, taint pass does not.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void reduction_sum(const float *input, float *output, int n)
{
    __shared__ float smem_partial[RED_N];

    int tid = threadIdx.x;
    smem_partial[tid] = (tid < n) ? input[blockIdx.x * RED_N + tid] : 0.0f;
    __syncthreads();

    for (int stride = RED_N / 2; stride > 0; stride >>= 1) {
        if (tid < stride)
            smem_partial[tid] += smem_partial[tid + stride];
        __syncthreads();
    }

    if (tid == 0)
        output[blockIdx.x] = smem_partial[0];

    // single exit
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 3: gemm_tile
// a_tile is loaded from a confidential matrix; b_tile is not.
// Conservative: flags both. Taint: flags only a_tile.
// ─────────────────────────────────────────────────────────────────────────────
__global__ void gemm_tile(
    __confidential__ const float *A,   // confidential weight matrix
    const float                  *B,
    float                        *C,
    int M, int N, int K)
{
    __shared__ float a_tile[GEMM_TILE * GEMM_TILE];   // tainted (loaded from A)
    __shared__ float b_tile[GEMM_TILE * GEMM_TILE];   // not tainted

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * GEMM_TILE + ty;
    int col = blockIdx.x * GEMM_TILE + tx;

    float acc = 0.0f;

    for (int t = 0; t < (K + GEMM_TILE - 1) / GEMM_TILE; t++) {
        // Load tiles into shared memory
        int a_col = t * GEMM_TILE + tx;
        int b_row = t * GEMM_TILE + ty;

        a_tile[ty * GEMM_TILE + tx] =
            (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        b_tile[ty * GEMM_TILE + tx] =
            (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < GEMM_TILE; k++)
            acc += a_tile[ty * GEMM_TILE + k] * b_tile[k * GEMM_TILE + tx];
        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;

    // single exit
}
