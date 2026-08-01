// benchmark_clock64.cu — RP §VI.D "On-device cycles" overhead measurement.
//
// Module 7 (benchmark.cu) uses host-side CUDA-event timing of the WHOLE
// kernel, which has a ~1-2 us launch-to-launch jitter floor that swallows
// the scrub cost (see results/benchmark.json -- all three classes reported
// as "< sigma" upper bounds, not resolved point estimates). This file
// instead brackets ONLY the scrub region with clock64() (jitter-free,
// on-device SM cycle counter), for all three kernel classes, matching
// benchmark.cu's problem sizes exactly so results are comparable.
//
// Method: __syncthreads() -> t0=clock64() -> [scrub region] -> t1=clock64()
// (the scrub region's own trailing __threadfence_block()+__syncthreads() is
// included in the bracket, matching what a real kernel exit pays). Only
// thread 0 records cycles, after a barrier guarantees all threads reached
// the same point -- avoids needing a cross-thread reduction for a value that
// should be near-identical across a warp/block on a single SM.
//
// Caveat (report honestly): clock64() counts SM clock cycles; GPU Boost /
// dynamic clocking means the SM clock at measurement time may differ from
// cudaDeviceProp.clockRate (queried once, at idle, before the runs). ns
// conversion uses the queried clockRate as a fixed reference; the resulting
// ns figure carries that source of imprecision on top of genuine cycle
// measurement (which itself is exact in cycles).
//
// Output: JSON to stdout -> results/benchmark_clock64.json

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_WARMUP   50
#define N_REPS    500     // >=500 reps per Phase 1.4 instructions

#define RED_N     4096
#define RED_BLOCK  256
#define GEMM_TILE   16
#define GEMM_K     256
#define ATTN_SEQ    64
#define ATTN_HEAD   32
#define ATTN_BLOCK  64

#define CUDA_CHECK(call)                                                      \
    do { cudaError_t _e=(call);                                               \
         if(_e!=cudaSuccess){                                                 \
             fprintf(stderr,"CUDA error %s:%d %s\n",                         \
                     __FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);}    \
    } while(0)

static int cmp_ll(const void *a,const void *b){
    long long fa=*(const long long*)a,fb=*(const long long*)b;
    return (fa>fb)-(fa<fb);
}
typedef struct { double med,std; } Stats;
static Stats compute_stats(long long *t,int n){
    long long *tmp=(long long*)malloc(n*sizeof(long long));
    for(int i=0;i<n;i++) tmp[i]=t[i];
    qsort(tmp,n,sizeof(long long),cmp_ll);
    double med=(n%2)?(double)tmp[n/2]:0.5*((double)tmp[n/2-1]+(double)tmp[n/2]);
    double mean=0; for(int i=0;i<n;i++) mean+=(double)t[i]; mean/=n;
    double v=0; for(int i=0;i<n;i++){double d=(double)t[i]-mean;v+=d*d;}
    v=(n>1)?v/(n-1):0;
    Stats s={med,sqrt(v)}; free(tmp); return s;
}

// ── REDUCTION: scrub region bracketed with clock64() ────────────────────────
__global__ void bench_reduction_scrub_clock(const float *in, float *out, long long *cycles_out) {
    __shared__ float smem[RED_N];
    int tid = threadIdx.x;
    float acc = 0.f;
    for (int i = tid; i < RED_N; i += RED_BLOCK) acc += in[i];
    smem[tid] = acc;
    __syncthreads();
    for (int s = RED_BLOCK/2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid+s];
        __syncthreads();
    }
    if (tid == 0) *out = smem[0];
    __syncthreads();
    long long t0 = clock64();
    volatile float *vp = smem;
    for (int i = tid; i < RED_N; i += RED_BLOCK) vp[i] = 0.f;
    __threadfence_block();
    __syncthreads();
    long long t1 = clock64();
    if (tid == 0) cycles_out[blockIdx.x] = t1 - t0;
}

// ── GEMM: scrub region bracketed with clock64() ──────────────────────────────
__global__ void bench_gemm_scrub_clock(const float *A, const float *B, float *C, long long *cycles_out) {
    __shared__ float a_tile[GEMM_TILE * GEMM_TILE];
    __shared__ float b_tile[GEMM_TILE * GEMM_TILE];
    int tid = threadIdx.x;
    int ty = tid / GEMM_TILE, tx = tid % GEMM_TILE;
    float acc = 0.f;
    for (int t = 0; t < GEMM_K / GEMM_TILE; t++) {
        a_tile[ty * GEMM_TILE + tx] = A[ty * GEMM_K + t * GEMM_TILE + tx];
        b_tile[ty * GEMM_TILE + tx] = B[(t * GEMM_TILE + ty) * GEMM_TILE + tx];
        __syncthreads();
        for (int k = 0; k < GEMM_TILE; k++)
            acc += a_tile[ty * GEMM_TILE + k] * b_tile[k * GEMM_TILE + tx];
        __syncthreads();
    }
    C[ty * GEMM_TILE + tx] = acc;
    __syncthreads();
    long long t0 = clock64();
    volatile float *vp = a_tile;
    for (int i = tid; i < GEMM_TILE * GEMM_TILE; i += blockDim.x) vp[i] = 0.f;
    __threadfence_block();
    __syncthreads();
    long long t1 = clock64();
    if (tid == 0) cycles_out[blockIdx.x] = t1 - t0;
}

// ── FULL-SM-FOOTPRINT AMPLIFICATION (RP Sec VI.D, item b) ───────────────────
// Scrubs a near-max shared-memory footprint (dynamic shared mem, since static
// __shared__ caps at 48 KB/block on sm_89) with a full-size block, to get a
// bandwidth-representative ns/word figure independent of any particular
// kernel's thread count/geometry.
#define AMPLIFY_WORDS   24000     // 24000 x 4B = 93.75 KB -- near the 100 KB/SM
                                   // ceiling on sm_89, leaving margin.
#define AMPLIFY_THREADS  1024     // max threads/block on sm_89
extern __shared__ float amp_buf[];
__global__ void bench_amplify_scrub_clock(long long *cycles_out) {
    int tid = threadIdx.x;
    for (int i = tid; i < AMPLIFY_WORDS; i += AMPLIFY_THREADS) amp_buf[i] = 1.f;
    __syncthreads();
    long long t0 = clock64();
    volatile float *vp = amp_buf;
    for (int i = tid; i < AMPLIFY_WORDS; i += AMPLIFY_THREADS) vp[i] = 0.f;
    __threadfence_block();
    __syncthreads();
    long long t1 = clock64();
    if (tid == 0) cycles_out[blockIdx.x] = t1 - t0;
}

// ── ATTENTION: scrub region bracketed with clock64() ─────────────────────────
__global__ void bench_attn_scrub_clock(const float *q, const float *k_in, float *out, long long *cycles_out) {
    __shared__ float smem_qk[ATTN_SEQ * ATTN_HEAD];
    __shared__ float smem_soft[ATTN_SEQ];
    int tid = threadIdx.x;
    for (int i = tid; i < ATTN_SEQ * ATTN_HEAD; i += ATTN_BLOCK)
        smem_qk[i] = q[i];
    __syncthreads();
    for (int row = tid; row < ATTN_SEQ; row += ATTN_BLOCK) {
        float s = 0.f;
        for (int j = 0; j < ATTN_HEAD; j++)
            s += smem_qk[row * ATTN_HEAD + j] * k_in[j];
        smem_soft[row] = s;
    }
    __syncthreads();
    for (int row = tid; row < ATTN_SEQ; row += ATTN_BLOCK)
        out[row] = smem_soft[row];
    __syncthreads();
    long long t0 = clock64();
    volatile float *vp = smem_qk;
    for (int i = tid; i < ATTN_SEQ * ATTN_HEAD; i += ATTN_BLOCK) vp[i] = 0.f;
    __threadfence_block();
    __syncthreads();
    long long t1 = clock64();
    if (tid == 0) cycles_out[blockIdx.x] = t1 - t0;
}

int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr,"[M7-clock64] %s (sm_%d%d), clockRate=%d kHz\n",
            prop.name,prop.major,prop.minor,prop.clockRate);

    float *d_ri,*d_ro, *d_A,*d_B,*d_C, *d_q,*d_k,*d_ao;
    long long *d_cycles;
    CUDA_CHECK(cudaMalloc(&d_ri, RED_N*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ro, sizeof(float)));
    { float *h=(float*)malloc(RED_N*4);
      for(int i=0;i<RED_N;i++) h[i]=1.f;
      CUDA_CHECK(cudaMemcpy(d_ri,h,RED_N*4,cudaMemcpyHostToDevice));
      free(h); }
    CUDA_CHECK(cudaMalloc(&d_A, GEMM_TILE*GEMM_K*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_B, GEMM_K*GEMM_TILE*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_C, GEMM_TILE*GEMM_TILE*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_A,0,GEMM_TILE*GEMM_K*4));
    CUDA_CHECK(cudaMemset(d_B,0,GEMM_K*GEMM_TILE*4));
    CUDA_CHECK(cudaMalloc(&d_q,  ATTN_SEQ*ATTN_HEAD*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_k,  ATTN_HEAD*sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ao, ATTN_SEQ*sizeof(float)));
    CUDA_CHECK(cudaMemset(d_q,0,ATTN_SEQ*ATTN_HEAD*4));
    CUDA_CHECK(cudaMemset(d_k,0,ATTN_HEAD*4));
    CUDA_CHECK(cudaMalloc(&d_cycles, sizeof(long long)));

    long long *h_cycles=(long long*)malloc(sizeof(long long));
    long long red_c[N_WARMUP+N_REPS], gemm_c[N_WARMUP+N_REPS], attn_c[N_WARMUP+N_REPS];

    fprintf(stderr,"[M7-clock64] reduction (%d warmup + %d reps) ...\n",N_WARMUP,N_REPS);
    for(int r=0;r<N_WARMUP+N_REPS;r++){
        bench_reduction_scrub_clock<<<1,RED_BLOCK>>>(d_ri,d_ro,d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_cycles,d_cycles,sizeof(long long),cudaMemcpyDeviceToHost));
        red_c[r]=*h_cycles;
    }

    fprintf(stderr,"[M7-clock64] gemm (%d warmup + %d reps) ...\n",N_WARMUP,N_REPS);
    for(int r=0;r<N_WARMUP+N_REPS;r++){
        bench_gemm_scrub_clock<<<1,GEMM_TILE*GEMM_TILE>>>(d_A,d_B,d_C,d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_cycles,d_cycles,sizeof(long long),cudaMemcpyDeviceToHost));
        gemm_c[r]=*h_cycles;
    }

    fprintf(stderr,"[M7-clock64] attention (%d warmup + %d reps) ...\n",N_WARMUP,N_REPS);
    for(int r=0;r<N_WARMUP+N_REPS;r++){
        bench_attn_scrub_clock<<<1,ATTN_BLOCK>>>(d_q,d_k,d_ao,d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_cycles,d_cycles,sizeof(long long),cudaMemcpyDeviceToHost));
        attn_c[r]=*h_cycles;
    }

    size_t amp_shmem = (size_t)AMPLIFY_WORDS*sizeof(float);
    CUDA_CHECK(cudaFuncSetAttribute(bench_amplify_scrub_clock,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)amp_shmem));
    long long amp_c[N_WARMUP+N_REPS];
    fprintf(stderr,"[M7-clock64] full-SM-footprint amplification: %d words (%.2f KB), %d threads (%d warmup + %d reps) ...\n",
            AMPLIFY_WORDS, amp_shmem/1024.0, AMPLIFY_THREADS, N_WARMUP, N_REPS);
    for(int r=0;r<N_WARMUP+N_REPS;r++){
        bench_amplify_scrub_clock<<<1,AMPLIFY_THREADS,amp_shmem>>>(d_cycles);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(h_cycles,d_cycles,sizeof(long long),cudaMemcpyDeviceToHost));
        amp_c[r]=*h_cycles;
    }

    Stats red_s  = compute_stats(red_c+N_WARMUP,N_REPS);
    Stats gemm_s = compute_stats(gemm_c+N_WARMUP,N_REPS);
    Stats attn_s = compute_stats(attn_c+N_WARMUP,N_REPS);
    Stats amp_s  = compute_stats(amp_c+N_WARMUP,N_REPS);

    double khz = (double)prop.clockRate; // clockRate is in kHz
    double red_ns  = red_s.med  * 1e6 / khz;
    double gemm_ns = gemm_s.med * 1e6 / khz;
    double attn_ns = attn_s.med * 1e6 / khz;
    double red_ns_std  = red_s.std  * 1e6 / khz;
    double gemm_ns_std = gemm_s.std * 1e6 / khz;
    double attn_ns_std = attn_s.std * 1e6 / khz;

    double amp_ns     = amp_s.med * 1e6 / khz;
    double amp_ns_std = amp_s.std * 1e6 / khz;
    double amp_ns_per_word = amp_ns / AMPLIFY_WORDS;
    // 95% CI on the mean via standard error (std/sqrt(n)) -- reported on the
    // ns/word figure since that's what Sec VI.D's placeholder asks for.
    double amp_sem_ns = amp_ns_std / sqrt((double)N_REPS);
    double amp_ci95_ns_per_word = 1.96 * amp_sem_ns / AMPLIFY_WORDS;

    fprintf(stderr,"reduction : %.1f +/- %.1f cycles = %.2f +/- %.2f ns\n",red_s.med,red_s.std,red_ns,red_ns_std);
    fprintf(stderr,"gemm      : %.1f +/- %.1f cycles = %.2f +/- %.2f ns\n",gemm_s.med,gemm_s.std,gemm_ns,gemm_ns_std);
    fprintf(stderr,"attention : %.1f +/- %.1f cycles = %.2f +/- %.2f ns\n",attn_s.med,attn_s.std,attn_ns,attn_ns_std);
    fprintf(stderr,"amplify   : %.1f +/- %.1f cycles = %.2f +/- %.2f ns total = %.6f ns/word (95%% CI +/- %.6f)\n",
            amp_s.med,amp_s.std,amp_ns,amp_ns_std,amp_ns_per_word,amp_ci95_ns_per_word);

    printf("{\n");
    printf("  \"experiment\": \"On-device clock64() scrub-region bracketing (RP Sec VI.D)\",\n");
    printf("  \"hardware\": \"%s (sm_%d%d)\",\n",prop.name,prop.major,prop.minor);
    printf("  \"gpu_clock_rate_khz\": %d,\n",prop.clockRate);
    printf("  \"caveat\": \"clock64() counts SM clock cycles exactly (jitter-free); ns conversion uses cudaDeviceProp.clockRate queried once at idle before the runs, so absolute ns figures inherit whatever imprecision GPU Boost/dynamic clocking introduces relative to that fixed reference. Cycle counts themselves are exact.\",\n");
    printf("  \"n_warmup\": %d,\n",N_WARMUP);
    printf("  \"n_reps\": %d,\n",N_REPS);
    printf("  \"method\": \"single-block kernel; __syncthreads() -> t0=clock64() -> [scrub loop + __threadfence_block() + __syncthreads()] -> t1=clock64(); thread 0 only; cycles = t1-t0; median/stddev over reps (post-warmup)\",\n");
    printf("  \"kernels\": [\n");
    printf("    {\"class\": \"reduction\",  \"scrub_words\": %d, \"cycles_median\": %.2f, \"cycles_stddev\": %.2f, \"ns_median\": %.4f, \"ns_stddev\": %.4f},\n",
           RED_N, red_s.med, red_s.std, red_ns, red_ns_std);
    printf("    {\"class\": \"gemm\",       \"scrub_words\": %d, \"cycles_median\": %.2f, \"cycles_stddev\": %.2f, \"ns_median\": %.4f, \"ns_stddev\": %.4f},\n",
           GEMM_TILE*GEMM_TILE, gemm_s.med, gemm_s.std, gemm_ns, gemm_ns_std);
    printf("    {\"class\": \"attention\",  \"scrub_words\": %d, \"cycles_median\": %.2f, \"cycles_stddev\": %.2f, \"ns_median\": %.4f, \"ns_stddev\": %.4f}\n",
           ATTN_SEQ*ATTN_HEAD, attn_s.med, attn_s.std, attn_ns, attn_ns_std);
    printf("  ],\n");
    printf("  \"full_sm_amplification\": {\n");
    printf("    \"description\": \"RP Sec VI.D item (b): scrubs a near-max shared-memory footprint (%d words, %.2f KB of ~100 KB/SM ceiling) with a full-size %d-thread block, dividing total scrub time by word count for a bandwidth-representative figure independent of any one kernel's thread geometry.\",\n",
           AMPLIFY_WORDS, (double)(AMPLIFY_WORDS*sizeof(float))/1024.0, AMPLIFY_THREADS);
    printf("    \"words\": %d,\n",AMPLIFY_WORDS);
    printf("    \"threads\": %d,\n",AMPLIFY_THREADS);
    printf("    \"cycles_median\": %.2f,\n",amp_s.med);
    printf("    \"cycles_stddev\": %.2f,\n",amp_s.std);
    printf("    \"total_ns_median\": %.4f,\n",amp_ns);
    printf("    \"ns_per_word\": %.6f,\n",amp_ns_per_word);
    printf("    \"ns_per_word_95ci\": %.6f\n",amp_ci95_ns_per_word);
    printf("  },\n");
    printf("  \"result\": \"Fills RP Table IV's clock64() column and Sec VI.D's [[MEASURE: cycles/exit per class]] / [[MEASURE: ns/exit]] placeholders with genuinely resolved, jitter-free on-device measurements, and the amplification method's [[MEASURE: ns per word, 95%% CI]] placeholder -- distinct from and more precise than benchmark.json's host-side CUDA-event upper bounds, which the scrub cost was too small to resolve against.\"\n");
    printf("}\n");

    free(h_cycles);
    cudaFree(d_ri); cudaFree(d_ro); cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_q); cudaFree(d_k); cudaFree(d_ao); cudaFree(d_cycles);
    return 0;
}
