// dse_minimal.cu — Bounded DSE probe kernels
//
// Attempt 1: Is ptxas eliminating dead shared-memory stores at all?
//
// Two probe pairs, each with a naive and a volatile version:
//
//   nosync_naive / nosync_volatile
//     Like the original scrub_naive but WITHOUT the __syncthreads() that
//     precedes the scrub.  Buffer is still read (in Phase 2) so Phase 1
//     stores are live, but Phase 3 scrub stores are dead.
//
//   pure_dead_naive / pure_dead_volatile
//     The absolute textbook dead store: buf is written in the scrub but
//     NEVER READ anywhere in the kernel.  Output is computed without
//     touching buf at all.  If ptxas eliminates any shared-store dead
//     stores, it must fire here.
//
// Compile three ways (see companion shell commands):
//   A: nvcc -arch=sm_89 -O3
//   B: nvcc -arch=sm_89 -O3 --use_fast_math
//   C: nvcc -arch=sm_89 -O3 -Xptxas -O3

#include <stdint.h>
#include <cuda_runtime.h>

#define M_NWORDS 1024
#define M_BLOCK   32

// ─────────────────────────────────────────────────────────────────────────────
// Probe 1a: nosync_naive
// Phase 1 stores are live (buf[0] read below).  Phase 3 scrub is dead.
// Removes the __syncthreads() that preceded the scrub in scrub_naive, so
// there is no barrier that could force live-ness of the scrub stores.
// ─────────────────────────────────────────────────────────────────────────────
__launch_bounds__(M_BLOCK, 1)
__global__ void nosync_naive(uint32_t *output) {
    __shared__ uint32_t buf[M_NWORDS];
    int tid = threadIdx.x;

    // Phase 1: live writes (buf[0] used below by thread 0)
    for (int i = tid; i < M_NWORDS; i += M_BLOCK)
        buf[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;

    // Phase 2: single read — no __syncthreads() before or after
    if (tid == 0) *output = buf[0] ^ buf[M_NWORDS - 1];

    // Phase 3: DEAD STORES — buf never read again, no barrier
    for (int i = tid; i < M_NWORDS; i += M_BLOCK)
        buf[i] = 0u;
}

// ─────────────────────────────────────────────────────────────────────────────
// Probe 1b: nosync_volatile  (same; volatile must prevent any elimination)
// ─────────────────────────────────────────────────────────────────────────────
__launch_bounds__(M_BLOCK, 1)
__global__ void nosync_volatile(uint32_t *output) {
    __shared__ uint32_t buf[M_NWORDS];
    int tid = threadIdx.x;

    for (int i = tid; i < M_NWORDS; i += M_BLOCK)
        buf[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;

    if (tid == 0) *output = buf[0] ^ buf[M_NWORDS - 1];

    volatile uint32_t *vbuf = buf;
    for (int i = tid; i < M_NWORDS; i += M_BLOCK)
        vbuf[i] = 0u;
}

// ─────────────────────────────────────────────────────────────────────────────
// Probe 2a: pure_dead_naive
// buf is NEVER READ anywhere in this kernel.  Output comes from tid only.
// These are the purest possible dead stores — if ptxas does ANY smem DSE
// at all, this is where it must fire.
// ─────────────────────────────────────────────────────────────────────────────
__launch_bounds__(M_BLOCK, 1)
__global__ void pure_dead_naive(uint32_t *output) {
    __shared__ uint32_t buf[M_NWORDS];
    int tid = threadIdx.x;

    // Output computed entirely independently of buf
    if (tid == 0) *output = 0u;

    // PURE DEAD STORES: buf written but never read — anywhere
    for (int i = tid; i < M_NWORDS; i += M_BLOCK)
        buf[i] = 0u;
}

// ─────────────────────────────────────────────────────────────────────────────
// Probe 2b: pure_dead_volatile  (volatile must survive even here)
// ─────────────────────────────────────────────────────────────────────────────
__launch_bounds__(M_BLOCK, 1)
__global__ void pure_dead_volatile(uint32_t *output) {
    __shared__ uint32_t buf[M_NWORDS];
    int tid = threadIdx.x;

    if (tid == 0) *output = 0u;

    volatile uint32_t *vbuf = buf;
    for (int i = tid; i < M_NWORDS; i += M_BLOCK)
        vbuf[i] = 0u;
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal host — exists only so nvcc emits a complete binary for cuobjdump.
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    uint32_t *d;
    cudaMalloc(&d, sizeof(uint32_t));
    nosync_naive       <<<1, M_BLOCK>>>(d);
    nosync_volatile    <<<1, M_BLOCK>>>(d);
    pure_dead_naive    <<<1, M_BLOCK>>>(d);
    pure_dead_volatile <<<1, M_BLOCK>>>(d);
    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}
