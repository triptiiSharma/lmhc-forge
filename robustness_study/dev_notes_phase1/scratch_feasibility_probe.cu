// scratch_feasibility_probe.cu — PLAN.md §9 feasibility compiles.
// Throwaway: validates Z6xK4 (inline-PTX scrub inside tiled-GEMM context)
// and Z2xK3 (volatile-no-barrier scrub inside reduction context) actually
// compile and produce the expected SASS shape before Phase 2 scaffolding.
// Not part of the study corpus; will be deleted after this check.

#include <stdint.h>
#include <cuda_runtime.h>

#define GEMM_TILE 16
#define RED_N 256

// ── Z6 x K4: inline-PTX hardened scrub inside a tiled-GEMM context ─────────
__launch_bounds__(GEMM_TILE * GEMM_TILE, 1)
__global__ void z6_k4_kernel(const float *A, const float *B, float *C,
                              int M, int N, int K) {
    __shared__ float a_tile[GEMM_TILE * GEMM_TILE];   // sensitive (scrub target)
    __shared__ float b_tile[GEMM_TILE * GEMM_TILE];   // not sensitive

    int tid = threadIdx.x;                // flat 0..255
    int tx  = tid % GEMM_TILE;
    int ty  = tid / GEMM_TILE;
    int row = blockIdx.y * GEMM_TILE + ty;
    int col = blockIdx.x * GEMM_TILE + tx;

    float acc = 0.0f;
    for (int t = 0; t < (K + GEMM_TILE - 1) / GEMM_TILE; t++) {
        int a_col = t * GEMM_TILE + tx;
        int b_row = t * GEMM_TILE + ty;
        a_tile[ty * GEMM_TILE + tx] = (row < M && a_col < K) ? A[row * K + a_col] : 0.0f;
        b_tile[ty * GEMM_TILE + tx] = (b_row < K && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < GEMM_TILE; k++)
            acc += a_tile[ty * GEMM_TILE + k] * b_tile[k * GEMM_TILE + tx];
        __syncthreads();
    }

    // Z6 scrub of a_tile — after last read, before the divergent output write.
    for (int i = tid; i < GEMM_TILE * GEMM_TILE; i += GEMM_TILE * GEMM_TILE) {
        unsigned int saddr = __cvta_generic_to_shared(&a_tile[i]);
        asm volatile("st.shared.u32 [%0], %1;" : : "r"(saddr), "r"(0u) : "memory");
    }
    __threadfence_block();
    __syncthreads();

    if (row < M && col < N)
        C[row * N + col] = acc;
}

// ── Z2 x K3: volatile zeroing loop, no fence, no trailing barrier, inside a
//    reduction context ──────────────────────────────────────────────────────
__launch_bounds__(RED_N, 1)
__global__ void z2_k3_kernel(const float *input, float *output, int n) {
    __shared__ float smem_partial[RED_N];   // sensitive (scrub target)
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

    // Z2 scrub: volatile, no fence, no additional barrier beyond what K3
    // already required above. Each thread writes only its own index (no
    // aliasing), so no race even without a trailing sync.
    volatile float *v = smem_partial;
    for (int i = tid; i < RED_N; i += RED_N)
        v[i] = 0.0f;
}

int main() {
    float *dA, *dB, *dC, *dIn, *dOut;
    cudaMalloc(&dA, sizeof(float));
    cudaMalloc(&dB, sizeof(float));
    cudaMalloc(&dC, sizeof(float));
    cudaMalloc(&dIn, sizeof(float));
    cudaMalloc(&dOut, sizeof(float));

    dim3 block4(GEMM_TILE * GEMM_TILE);
    z6_k4_kernel<<<1, block4>>>(dA, dB, dC, 16, 16, 16);
    z2_k3_kernel<<<1, RED_N>>>(dIn, dOut, RED_N);
    cudaDeviceSynchronize();

    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dIn); cudaFree(dOut);
    return 0;
}
