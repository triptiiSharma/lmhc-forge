// dse_test.cu — Module 3: Dead-Store Elimination Differential
//
// Compiles TWO versions of an identical computation kernel that differ ONLY
// in how they scrub __shared__ memory before returning.
//
// (a) scrub_naive   — plain `buf[i] = 0` loop.  At -O3, LLVM/ptxas may
//                     dead-store-eliminate these writes because buf is never
//                     read after the scrub and __shared__ SRAM is invisible
//                     outside the block.
//
// (b) scrub_hardened — volatile pointer + __threadfence_block().  volatile
//                     writes are observable side-effects in the C++ memory
//                     model and CANNOT be eliminated.
//
// Compile at -O3 (not -O2) to trigger aggressive DSE:
//   nvcc -arch=sm_89 -O3 -ccbin gcc-14 -o dse_test dse_test.cu
//
// Inspect with cuobjdump:
//   cuobjdump -sass dse_test | grep -E "(Function|STS)"
//
// The machine-checked store count in sass_dumps/ is the Theorem 1 evidence.

#include <stdint.h>
#include <cuda_runtime.h>

// Shared buffer size chosen to be large enough that a simple unroll cannot
// replace it with register operations, but small enough that SASS is readable.
#define DSE_NWORDS 1024
#define DSE_BLOCK   32

// ─────────────────────────────────────────────────────────────────────────────
// Kernel (a): NAIVE SCRUB
// Writes buf[i] = 0 with a plain pointer.  The compiler is free to observe
// that buf is not read after this point and eliminate the stores entirely.
// ─────────────────────────────────────────────────────────────────────────────
__launch_bounds__(DSE_BLOCK, 1)
__global__ void scrub_naive(uint32_t *output) {
    __shared__ uint32_t buf[DSE_NWORDS];
    int tid = threadIdx.x;

    // Phase 1 — simulate sensitive computation (writes to buf)
    for (int i = tid; i < DSE_NWORDS; i += DSE_BLOCK)
        buf[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;
    __syncthreads();

    // Phase 2 — produce output (reads from buf)
    if (tid == 0)
        *output = buf[0] ^ buf[DSE_NWORDS - 1];
    __syncthreads();

    // Phase 3 — NAIVE SCRUB: after this, buf is never read again.
    // At -O3, the compiler can legally remove these stores (dead-store elim).
    for (int i = tid; i < DSE_NWORDS; i += DSE_BLOCK)
        buf[i] = 0u;
    // ← kernel returns; no further use of buf.
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel (b): HARDENED SCRUB
// Identical Phase 1 and Phase 2.  Phase 3 uses a volatile pointer so that
// the C++ abstract machine MUST execute every store, and __threadfence_block()
// orders the stores before the implicit block-termination fence.
// ─────────────────────────────────────────────────────────────────────────────
__launch_bounds__(DSE_BLOCK, 1)
__global__ void scrub_hardened(uint32_t *output) {
    __shared__ uint32_t buf[DSE_NWORDS];
    int tid = threadIdx.x;

    // Phase 1 — identical sensitive computation
    for (int i = tid; i < DSE_NWORDS; i += DSE_BLOCK)
        buf[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;
    __syncthreads();

    // Phase 2 — identical output
    if (tid == 0)
        *output = buf[0] ^ buf[DSE_NWORDS - 1];
    __syncthreads();

    // Phase 3 — HARDENED SCRUB: volatile prevents DSE; fence orders stores.
    volatile uint32_t *vbuf = buf;
    for (int i = tid; i < DSE_NWORDS; i += DSE_BLOCK)
        vbuf[i] = 0u;
    __threadfence_block();
    __syncthreads();
    // ← kernel returns; volatile stores guaranteed to have occurred.
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel (c): PTX INLINE-ASM FALLBACK (if volatile+fence is insufficient on Ada)
// Uses inline PTX st.shared which ptxas cannot eliminate at any opt level.
// ─────────────────────────────────────────────────────────────────────────────
__launch_bounds__(DSE_BLOCK, 1)
__global__ void scrub_ptx_fallback(uint32_t *output) {
    __shared__ uint32_t buf[DSE_NWORDS];
    int tid = threadIdx.x;

    // Phase 1
    for (int i = tid; i < DSE_NWORDS; i += DSE_BLOCK)
        buf[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;
    __syncthreads();

    // Phase 2
    if (tid == 0)
        *output = buf[0] ^ buf[DSE_NWORDS - 1];
    __syncthreads();

    // Phase 3 — PTX inline-asm: direct st.shared.u32; ptxas cannot remove this.
    // __cvta_generic_to_shared converts the generic 64-bit pointer to the 32-bit
    // shared-memory-space address that st.shared.u32 expects.
    for (int i = tid; i < DSE_NWORDS; i += DSE_BLOCK) {
        unsigned int saddr = __cvta_generic_to_shared(buf + i);
        asm volatile("st.shared.u32 [%0], %1;" : : "r"(saddr), "r"(0u) : "memory");
    }
    __threadfence_block();
    __syncthreads();
}

// ─────────────────────────────────────────────────────────────────────────────
// Minimal host — only exists so nvcc produces a complete binary for cuobjdump.
// The actual analysis is done by dse_analysis.py reading the SASS dump.
// ─────────────────────────────────────────────────────────────────────────────
int main() {
    uint32_t *d_out;
    cudaMalloc(&d_out, sizeof(uint32_t));

    scrub_naive       <<<1, DSE_BLOCK>>>(d_out);
    scrub_hardened    <<<1, DSE_BLOCK>>>(d_out);
    scrub_ptx_fallback<<<1, DSE_BLOCK>>>(d_out);
    cudaDeviceSynchronize();

    cudaFree(d_out);
    return 0;
}
