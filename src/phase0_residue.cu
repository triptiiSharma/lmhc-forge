// phase0_residue.cu — Phase 0: Cross-Launch Shared-Memory Residue Detector
// LMHC-Forge evaluation artifact
// Hardware target: NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, sm_89)
//
// Design:
//   Kernel A  — writes a known sentinel pattern into __shared__ memory, exits WITHOUT
//               zeroing.  Low occupancy (__launch_bounds__), single block, one warp.
//   Kernel B  — declares an identically-sized __shared__ buffer, reads it BEFORE any
//               write.  Records which words are non-zero (potential residue from A).
//   Both kernels expose their SM id so we can split results into same-SM / diff-SM.
//   N = 1000 trials.  Output: residue rate = fraction of non-zero words.

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ---- tunables ---------------------------------------------------------------
#define NWORDS      4096           // 4096 x uint32 = 16 KB shared memory
#define NTRIALS     1000
#define BLOCK_DIM   32             // one warp — minimum meaningful block
#define PATTERN     0xCAFEBABEu    // sentinel written by Kernel A
// -----------------------------------------------------------------------------

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t _e = (call);                                                  \
        if (_e != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error at %s:%d  %s\n",                         \
                    __FILE__, __LINE__, cudaGetErrorString(_e));                  \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// PTX: read streaming-multiprocessor id
__device__ __forceinline__ uint32_t get_smid() {
    uint32_t id;
    asm volatile("mov.u32 %0, %smid;" : "=r"(id));
    return id;
}

// ---------------------------------------------------------------------------
// Kernel A — WRITER
//   Fills __shared__ buf[NWORDS] with PATTERN.  Exits without zeroing.
//   __launch_bounds__(BLOCK_DIM, 1): compiler constrains register bank to allow
//   ≥1 block/SM; combined with 16 KB shared footprint this keeps occupancy low.
// ---------------------------------------------------------------------------
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelA(uint32_t *sm_out) {
    __shared__ uint32_t buf[NWORDS];
    const int tid = threadIdx.x;

    if (tid == 0) *sm_out = get_smid();

    for (int i = tid; i < NWORDS; i += BLOCK_DIM)
        buf[i] = PATTERN;

    __syncthreads();

    // Volatile sink: prevents the compiler from dead-store-eliminating the
    // writes above (the whole point is that the pattern stays in SRAM).
    if (tid == 0) {
        volatile uint32_t sink = buf[NWORDS - 1];
        (void)sink;
    }
    // ---- EXIT WITHOUT SCRUB ----
}

// ---------------------------------------------------------------------------
// Kernel B — READER (cross-launch residue check)
//   Reads __shared__ buf[NWORDS] BEFORE writing anything.  Counts non-zero
//   words and saves the first BLOCK_DIM values for diagnostic display.
// ---------------------------------------------------------------------------
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelB(uint32_t *sm_out, int *nz_out, uint32_t *sample_out) {
    __shared__ uint32_t buf[NWORDS];
    const int tid = threadIdx.x;

    if (tid == 0) *sm_out = get_smid();

    // === READ BEFORE ANY WRITE — cross-launch residue measurement ============
    int nz = 0;
    for (int i = tid; i < NWORDS; i += BLOCK_DIM) {
        uint32_t val = buf[i];      // uninitialized — may hold A's residue
        if (val != 0u) nz++;
        if (i < BLOCK_DIM)          // save first warp-worth of words
            sample_out[i] = val;
    }
    // =========================================================================

    // Warp reduction (full-warp mask, BLOCK_DIM == 32)
    for (int off = 16; off > 0; off >>= 1)
        nz += __shfl_down_sync(0xFFFFFFFFu, nz, off);

    if (tid == 0)
        atomicAdd(nz_out, nz);
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
int main() {
    // Print device info
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("=== Phase 0: Cross-Launch Shared-Memory Residue Detector ===\n");
    printf("Hardware : %s  (sm_%d%d)\n",
           prop.name, prop.major, prop.minor);
    printf("Shared/SM: %zu KB max\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("SMs      : %d\n", prop.multiProcessorCount);
    printf("Buffer   : %d x uint32 = %d KB\n", NWORDS, NWORDS * 4 / 1024);
    printf("Pattern  : 0x%08X\n", PATTERN);
    printf("Trials   : %d\n", NTRIALS);
    printf("-----------------------------------------------------------\n");

    uint32_t *d_sm_a, *d_sm_b, *d_sample;
    int      *d_nz;
    CUDA_CHECK(cudaMalloc(&d_sm_a,   sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_sm_b,   sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_nz,     sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sample, BLOCK_DIM * sizeof(uint32_t)));

    uint64_t total_nz_same = 0;
    uint64_t total_nz_all  = 0;
    int same_sm = 0, diff_sm = 0;

    for (int t = 0; t < NTRIALS; t++) {
        CUDA_CHECK(cudaMemset(d_nz, 0, sizeof(int)));

        // --- Kernel A (writer) ---
        kernelA<<<1, BLOCK_DIM>>>(d_sm_a);
        CUDA_CHECK(cudaDeviceSynchronize());

        // --- Kernel B (reader) ---
        kernelB<<<1, BLOCK_DIM>>>(d_sm_b, d_nz, d_sample);
        CUDA_CHECK(cudaDeviceSynchronize());

        uint32_t h_sm_a, h_sm_b;
        int h_nz;
        CUDA_CHECK(cudaMemcpy(&h_sm_a, d_sm_a, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_sm_b, d_sm_b, sizeof(uint32_t), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&h_nz,   d_nz,   sizeof(int),      cudaMemcpyDeviceToHost));

        total_nz_all += (uint64_t)h_nz;
        if (h_sm_a == h_sm_b) {
            same_sm++;
            total_nz_same += (uint64_t)h_nz;
        } else {
            diff_sm++;
        }
    }

    // Grab sample from last trial
    uint32_t h_sample[BLOCK_DIM];
    CUDA_CHECK(cudaMemcpy(h_sample, d_sample,
                          BLOCK_DIM * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // --- Results ---
    printf("Same-SM trials : %d / %d\n", same_sm, NTRIALS);
    printf("Diff-SM trials : %d / %d\n", diff_sm, NTRIALS);
    printf("-----------------------------------------------------------\n");

    // Same-SM residue rate (most meaningful: A and B shared the same physical SRAM)
    if (same_sm > 0) {
        uint64_t denom_same = (uint64_t)same_sm * NWORDS;
        double rr_same = (double)total_nz_same / (double)denom_same;
        printf("Non-zero words (same-SM) : %llu / %llu\n",
               (unsigned long long)total_nz_same,
               (unsigned long long)denom_same);
        printf("Residue rate  (same-SM)  : %.6f  (%.4f%%)\n",
               rr_same, rr_same * 100.0);
    } else {
        printf("(No same-SM trials recorded — both kernels always landed on different SMs.)\n");
    }

    // All-trials residue rate (conservative upper bound)
    {
        uint64_t denom_all = (uint64_t)NTRIALS * NWORDS;
        double rr_all = (double)total_nz_all / (double)denom_all;
        printf("Non-zero words (all)     : %llu / %llu\n",
               (unsigned long long)total_nz_all,
               (unsigned long long)denom_all);
        printf("Residue rate  (all)      : %.6f  (%.4f%%)\n",
               rr_all, rr_all * 100.0);
    }

    // Sample dump — diagnostic
    printf("-----------------------------------------------------------\n");
    printf("Last-trial sample: first %d words read by Kernel B (before any write):\n",
           BLOCK_DIM);
    int pattern_hits = 0;
    for (int i = 0; i < BLOCK_DIM; i++) {
        int is_pattern = (h_sample[i] == PATTERN);
        pattern_hits += is_pattern;
        printf("  buf[%2d] = 0x%08X%s\n", i, h_sample[i],
               is_pattern ? "  <-- PATTERN" : "");
    }
    printf("Pattern hits in sample: %d / %d\n", pattern_hits, BLOCK_DIM);

    cudaFree(d_sm_a);
    cudaFree(d_sm_b);
    cudaFree(d_nz);
    cudaFree(d_sample);
    return 0;
}
