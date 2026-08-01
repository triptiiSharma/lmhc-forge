// power_sustained.cu — RP Sec VI.D "Energy and thermal cost": sustained-load
// driver for nvidia-smi power.draw sampling. Runs one of two modes for a
// fixed wall-clock duration, queuing many kernel launches so the GPU stays
// busy long enough for nvidia-smi's ~100 ms sampling granularity to observe
// a difference, if one exists.
//
//   mode=idle    : device left idle (process alive, no kernel launches) --
//                  baseline power draw of this GPU under WSL2 at rest.
//   mode=empty   : repeated launches of a kernel that fills shared memory
//                  but performs NO scrub (fill only).
//   mode=scrub   : identical fill, PLUS the full LMHC hardened scrub loop
//                  (same footprint as benchmark_clock64.cu's amplification
//                  kernel: ~93.75 KB, 1024 threads).
//
// Usage: power_sustained <idle|empty|scrub> <seconds>
// Prints nothing GPU-side; a wrapper script samples nvidia-smi in parallel.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <cuda_runtime.h>

#define AMPLIFY_WORDS 24000
#define AMPLIFY_THREADS 1024

extern __shared__ float amp_buf[];

__global__ void fill_only() {
    int tid = threadIdx.x;
    for (int i = tid; i < AMPLIFY_WORDS; i += AMPLIFY_THREADS) amp_buf[i] = 1.f;
    __syncthreads();
    volatile float *vp = amp_buf;
    if (tid == 0) { float x = vp[0]; (void)x; }
}

__global__ void fill_and_scrub() {
    int tid = threadIdx.x;
    for (int i = tid; i < AMPLIFY_WORDS; i += AMPLIFY_THREADS) amp_buf[i] = 1.f;
    __syncthreads();
    volatile float *vp = amp_buf;
    for (int i = tid; i < AMPLIFY_WORDS; i += AMPLIFY_THREADS) vp[i] = 0.f;
    __threadfence_block();
    __syncthreads();
}

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

int main(int argc, char **argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <idle|empty|scrub> <seconds>\n", argv[0]); return 1; }
    const char *mode = argv[1];
    double duration = atof(argv[2]);

    size_t shmem = (size_t)AMPLIFY_WORDS * sizeof(float);
    if (strcmp(mode, "idle") == 0) {
        double t0 = now_s();
        while (now_s() - t0 < duration) { /* spin, no GPU work */ }
        return 0;
    }

    cudaFuncSetAttribute(fill_only, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);
    cudaFuncSetAttribute(fill_and_scrub, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shmem);

    double t0 = now_s();
    long launches = 0;
    while (now_s() - t0 < duration) {
        for (int i = 0; i < 200; i++) {
            if (strcmp(mode, "scrub") == 0)
                fill_and_scrub<<<1, AMPLIFY_THREADS, shmem>>>();
            else
                fill_only<<<1, AMPLIFY_THREADS, shmem>>>();
        }
        cudaDeviceSynchronize();
        launches += 200;
    }
    fprintf(stderr, "[power_sustained] mode=%s launches=%ld duration=%.2fs\n", mode, launches, now_s() - t0);
    return 0;
}
