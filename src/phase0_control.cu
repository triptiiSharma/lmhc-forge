// phase0_control.cu — Control experiment for Phase 0
// Kernel A is NEVER launched in this process. Only Kernel B runs, reading
// its own uninitialized __shared__ buffer. Rules out self-contamination.
//
// Shared memory layout, buffer size, block size, SM-ID tracking, and
// residue counting are identical to phase0_residue.cu so the comparison
// is apples-to-apples.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NWORDS    4096
#define NTRIALS   1000
#define BLOCK_DIM 32
#define PATTERN   0xCAFEBABEu

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t _e = (call);                                            \
        if (_e != cudaSuccess) {                                            \
            fprintf(stderr, "CUDA error %d at %s:%d  %s\n",                \
                    (int)_e, __FILE__, __LINE__, cudaGetErrorString(_e));   \
            exit(1);                                                        \
        }                                                                   \
    } while (0)

__device__ __forceinline__ uint32_t get_smid() {
    uint32_t id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(id));
    return id;
}

// Kernel B — identical to phase0_residue.cu: reads shared buffer before
// writing. Kernel A has NEVER run in this process.
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelB_only(uint32_t *sm_out, int *nz_out,
                              int *pattern_out, uint32_t *sample_out) {
    __shared__ uint32_t buf[NWORDS];
    const int tid = threadIdx.x;

    if (tid == 0) *sm_out = get_smid();

    // READ BEFORE ANY WRITE — no Kernel A has ever touched this buffer
    int nz      = 0;
    int pattern = 0;
    for (int i = tid; i < NWORDS; i += BLOCK_DIM) {
        uint32_t val = buf[i];
        if (val != 0u)       nz++;
        if (val == PATTERN)  pattern++;
        if (i < BLOCK_DIM)   sample_out[i] = val;
    }

    for (int off = 16; off > 0; off >>= 1) {
        nz      += __shfl_down_sync(0xFFFFFFFFu, nz,      off);
        pattern += __shfl_down_sync(0xFFFFFFFFu, pattern, off);
    }

    if (tid == 0) {
        atomicAdd(nz_out,      nz);
        atomicAdd(pattern_out, pattern);
    }
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Phase 0 CONTROL: Kernel-B-Only (Kernel A Never Launched) ===\n");
    printf("Hardware : %s  (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    printf("Buffer   : %d x uint32 = %d KB\n", NWORDS, NWORDS * 4 / 1024);
    printf("Pattern  : 0x%08X  (sentinel — Kernel A never wrote it)\n", PATTERN);
    printf("Trials   : %d\n", NTRIALS);
    printf("-----------------------------------------------------------\n");

    uint32_t *d_sm, *d_sample;
    int      *d_nz, *d_pattern;
    CUDA_CHECK(cudaMalloc(&d_sm,      sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sample,  BLOCK_DIM * sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_nz,      sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pattern, sizeof(int)));

    uint64_t total_nz      = 0;
    uint64_t total_pattern = 0;

    for (int t = 0; t < NTRIALS; t++) {
        CUDA_CHECK(cudaMemset(d_nz,      0, sizeof(int)));
        CUDA_CHECK(cudaMemset(d_pattern, 0, sizeof(int)));

        // ---- Kernel A is intentionally absent ----
        kernelB_only<<<1, BLOCK_DIM>>>(d_sm, d_nz, d_pattern, d_sample);
        CUDA_CHECK(cudaDeviceSynchronize());

        int h_nz, h_pattern;
        CUDA_CHECK(cudaMemcpy(&h_nz,      d_nz,      sizeof(int), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_pattern, d_pattern, sizeof(int), cudaMemcpyDeviceToHost));
        total_nz      += (uint64_t)h_nz;
        total_pattern += (uint64_t)h_pattern;
    }

    uint32_t h_sample[BLOCK_DIM];
    CUDA_CHECK(cudaMemcpy(h_sample, d_sample,
                          BLOCK_DIM * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    uint64_t denom = (uint64_t)NTRIALS * NWORDS;
    double rr      = (double)total_nz      / (double)denom;
    double pr      = (double)total_pattern / (double)denom;

    printf("Non-zero words        : %llu / %llu\n",
           (unsigned long long)total_nz, (unsigned long long)denom);
    printf("Residue rate          : %.6f  (%.4f%%)\n", rr, rr * 100.0);
    printf("0xCAFEBABE hits       : %llu / %llu\n",
           (unsigned long long)total_pattern, (unsigned long long)denom);
    printf("Pattern hit rate      : %.6f  (%.4f%%)\n", pr, pr * 100.0);
    printf("-----------------------------------------------------------\n");
    printf("Last-trial sample (first %d words, Kernel B, no prior Kernel A):\n",
           BLOCK_DIM);
    for (int i = 0; i < BLOCK_DIM; i++) {
        int is_pattern = (h_sample[i] == PATTERN);
        printf("  buf[%2d] = 0x%08X%s\n", i, h_sample[i],
               is_pattern ? "  <-- PATTERN (unexpected!)" : "");
    }

    cudaFree(d_sm); cudaFree(d_sample);
    cudaFree(d_nz); cudaFree(d_pattern);
    return 0;
}
