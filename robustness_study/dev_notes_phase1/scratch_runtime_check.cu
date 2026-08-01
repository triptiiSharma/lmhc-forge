// scratch_runtime_check.cu — empirical completeness check for the z2_k3
// scrub-placement pattern (PLAN.md §9). Static SASS inspection showed
// multiple predicated EXIT instructions and a STS...RZ count that doesn't
// map cleanly onto "256 words", which is expected once STS is understood as
// a per-warp SIMT instruction (each static STS executes once per warp that
// reaches it, writing up to 32 words) rather than a per-word instruction.
// Rather than reason about that further from text alone, this actually runs
// the kernel on the RTX 4050 (sm_89) and reads back every word of the
// shared buffer AFTER the scrub, via global memory, to settle empirically
// whether every one of the 256 threads' zero-stores actually executed.
// Throwaway — not part of the study corpus.

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define RED_N 256

__launch_bounds__(RED_N, 1)
__global__ void z2_k3_diag(const float *input, float *output,
                            float *post_scrub_dump, int n) {
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

    // Z2 scrub: volatile, no fence, no additional barrier.
    volatile float *v = smem_partial;
    for (int i = tid; i < RED_N; i += RED_N)
        v[i] = 0.0f;

    // Diagnostic only (not part of the Z2 construct definition): every
    // thread reads back its own word post-scrub to global memory so the
    // host can verify completeness. Placed after the scrub with no
    // synchronization change relative to the construct under test.
    post_scrub_dump[tid] = smem_partial[tid];
}

int main() {
    float h_in[RED_N];
    for (int i = 0; i < RED_N; i++) h_in[i] = (float)(i + 1);

    float *d_in, *d_out, *d_dump;
    cudaMalloc(&d_in, RED_N * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMalloc(&d_dump, RED_N * sizeof(float));
    cudaMemcpy(d_in, h_in, RED_N * sizeof(float), cudaMemcpyHostToDevice);

    z2_k3_diag<<<1, RED_N>>>(d_in, d_out, d_dump, RED_N);
    cudaError_t launch_err = cudaGetLastError();
    cudaError_t sync_err   = cudaDeviceSynchronize();

    float h_out, h_dump[RED_N];
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_dump, d_dump, RED_N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("launch_err=%s sync_err=%s\n",
           cudaGetErrorString(launch_err), cudaGetErrorString(sync_err));

    float expected_sum = 0.0f;
    for (int i = 0; i < RED_N; i++) expected_sum += h_in[i];
    printf("reduction output = %f (expected %f) -> %s\n",
           h_out, expected_sum,
           (h_out == expected_sum) ? "MATCH" : "MISMATCH");

    int nonzero_count = 0;
    for (int i = 0; i < RED_N; i++) {
        if (h_dump[i] != 0.0f) {
            nonzero_count++;
            if (nonzero_count <= 5)
                printf("  NONZERO after scrub: post_scrub_dump[%d] = %f\n", i, h_dump[i]);
        }
    }
    printf("words nonzero after scrub: %d / %d -> %s\n",
           nonzero_count, RED_N,
           (nonzero_count == 0) ? "ALL 256 THREADS SCRUBBED CORRECTLY"
                                 : "SOME THREADS SKIPPED THEIR SCRUB STORE (BUG)");

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_dump);
    return 0;
}
