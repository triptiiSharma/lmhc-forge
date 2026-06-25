// benchmark.cu — Module 7: LMHC Benchmark Suite
// Measures baseline vs LMHC-scrubbed latency for three kernel classes.
// Hardware: NVIDIA GeForce RTX 4050 Laptop GPU (Ada Lovelace, sm_89)
// Output: JSON to stdout  →  results/benchmark.json
//
// Timing method: interleaved batch.
//   Each rep: N_BATCH baseline launches, then N_BATCH scrubbed launches in the
//   SAME stream with three CUDA events.  Per-rep overhead = (scrubbed_half -
//   baseline_half) / N_BATCH.  Median and stddev are computed from the
//   N_REPS per-rep overhead values.
//
//   If |median_overhead| < overhead_stddev the result is flagged
//   "below_noise_floor" — reported as an upper bound (< σ), NOT as a negative %.
//
//   Scrub-isolation runs for kernels whose overhead is below the noise floor:
//   an empty kernel (same shared-memory allocation) is timed vs a scrub-only
//   kernel.  Uses N_ISO_BATCH=1000, N_ISO_REPS=80 to push the noise floor down.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define N_GLOBAL_WARMUP  500
#define N_WARMUP          10
#define N_REPS            50
#define N_BATCH          200

#define N_ISO_WARMUP      20
#define N_ISO_REPS        80
#define N_ISO_BATCH     1000

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

static int cmp_f(const void *a,const void *b){
    float fa=*(const float*)a,fb=*(const float*)b;
    return (fa>fb)-(fa<fb);
}
typedef struct { float med,std; } Stats;
static Stats compute_stats(float *t,int n){
    qsort(t,n,sizeof(float),cmp_f);
    float med=(n%2)?t[n/2]:0.5f*(t[n/2-1]+t[n/2]);
    float mean=0; for(int i=0;i<n;i++) mean+=t[i]; mean/=n;
    float v=0; for(int i=0;i<n;i++){float d=t[i]-mean;v+=d*d;}
    v=(n>1)?v/(n-1):0;
    Stats s={med,sqrtf(v)}; return s;
}

// ── REDUCTION ────────────────────────────────────────────────────────────────

__global__ void bench_reduction_baseline(const float *in, float *out) {
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
}

__global__ void bench_reduction_scrubbed(const float *in, float *out) {
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
    volatile float *vp = smem;
    for (int i = tid; i < RED_N; i += RED_BLOCK) vp[i] = 0.f;
    __threadfence_block();
    __syncthreads();
}

__global__ void bench_red_empty() {
    __shared__ float smem[RED_N];
    volatile float *vp = smem;
    if (threadIdx.x == 0) { float x = vp[0]; (void)x; }
}

__global__ void bench_red_scrub_only() {
    __shared__ float smem[RED_N];
    volatile float *vp = smem;
    int tid = threadIdx.x;
    for (int i = tid; i < RED_N; i += RED_BLOCK) vp[i] = 0.f;
    __threadfence_block();
    __syncthreads();
}

// ── GEMM ─────────────────────────────────────────────────────────────────────

__global__ void bench_gemm_baseline(const float *A, const float *B, float *C) {
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
}

__global__ void bench_gemm_scrubbed(const float *A, const float *B, float *C) {
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
    volatile float *vp = a_tile;
    for (int i = tid; i < GEMM_TILE * GEMM_TILE; i += blockDim.x) vp[i] = 0.f;
    __threadfence_block();
    __syncthreads();
}

// ── ATTENTION ────────────────────────────────────────────────────────────────

__global__ void bench_attn_baseline(const float *q, const float *k_in, float *out) {
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
}

__global__ void bench_attn_scrubbed(const float *q, const float *k_in, float *out) {
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
    volatile float *vp = smem_qk;
    for (int i = tid; i < ATTN_SEQ * ATTN_HEAD; i += ATTN_BLOCK) vp[i] = 0.f;
    __threadfence_block();
    __syncthreads();
}

__global__ void bench_attn_empty() {
    __shared__ float smem_qk[ATTN_SEQ * ATTN_HEAD];
    __shared__ float smem_soft[ATTN_SEQ];
    volatile float *vp  = smem_qk;
    volatile float *vp2 = smem_soft;
    if (threadIdx.x == 0) { float x = vp[0] + vp2[0]; (void)x; }
}

__global__ void bench_attn_scrub_only() {
    __shared__ float smem_qk[ATTN_SEQ * ATTN_HEAD];
    __shared__ float smem_soft[ATTN_SEQ];
    volatile float *vp = smem_qk;
    int tid = threadIdx.x;
    for (int i = tid; i < ATTN_SEQ * ATTN_HEAD; i += ATTN_BLOCK) vp[i] = 0.f;
    __threadfence_block();
    __syncthreads();
    (void)smem_soft;
}

// ── JSON helpers ──────────────────────────────────────────────────────────────

static void print_kernel_json(
        const char *label, const char *desc,
        float shared_kb, int scrub_words, int block_threads,
        Stats base, Stats scrub, Stats ovhd,
        int below_floor,
        Stats iso_ovhd, Stats iso_base, Stats iso_scrub, int has_iso,
        float occ, const char *occ_note,
        int last)
{
    float pct = (base.med > 0.f) ? 100.f * ovhd.med / base.med : 0.f;
    printf("    {\n");
    printf("      \"class\": \"%s\",\n", label);
    printf("      \"description\": \"%s\",\n", desc);
    printf("      \"shared_kb\": %.2f,\n", shared_kb);
    printf("      \"scrub_words\": %d,\n", scrub_words);
    printf("      \"block_threads\": %d,\n", block_threads);
    printf("      \"baseline_median_us\": %.4f,\n", base.med);
    printf("      \"baseline_stddev_us\": %.4f,\n", base.std);
    printf("      \"scrubbed_median_us\": %.4f,\n", scrub.med);
    printf("      \"scrubbed_stddev_us\": %.4f,\n", scrub.std);
    printf("      \"overhead_median_us\": %.4f,\n", ovhd.med);
    printf("      \"overhead_stddev_us\": %.4f,\n", ovhd.std);
    printf("      \"overhead_pct\": %.2f,\n", pct);
    printf("      \"below_noise_floor\": %s,\n", below_floor ? "true" : "false");
    if (below_floor) {
        printf("      \"upper_bound_us\": %.4f,\n", ovhd.std);
        printf("      \"noise_floor_note\": \"median overhead (%.4f us) < stddev (%.4f us); bounded as scrub cost < %.4f us (1 sigma)\",\n",
               ovhd.med, ovhd.std, ovhd.std);
    }
    if (has_iso) {
        float iso_pct = (base.med > 0.f) ? 100.f * iso_ovhd.med / base.med : 0.f;
        printf("      \"scrub_isolated_overhead_us\": %.4f,\n", iso_ovhd.med);
        printf("      \"scrub_isolated_stddev_us\": %.4f,\n",   iso_ovhd.std);
        printf("      \"scrub_isolated_empty_us\": %.4f,\n",    iso_base.med);
        printf("      \"scrub_isolated_scrub_only_us\": %.4f,\n", iso_scrub.med);
        printf("      \"scrub_isolated_pct_of_baseline\": %.2f,\n", iso_pct);
    }
    printf("      \"occupancy\": %.4f,\n", occ);
    printf("      \"occupancy_note\": \"%s\"\n", occ_note);
    printf("    }%s\n", last ? "" : ",");
}

// ── main ─────────────────────────────────────────────────────────────────────

int main(void) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    fprintf(stderr,"[M7] %s (sm_%d%d)\n",prop.name,prop.major,prop.minor);

    float *d_ri,*d_ro, *d_A,*d_B,*d_C, *d_q,*d_k,*d_ao;
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

    fprintf(stderr,"[M7] global warmup (%d × 6 kernels) ...\n",N_GLOBAL_WARMUP);
    for(int i=0;i<N_GLOBAL_WARMUP;i++){
        bench_reduction_baseline<<<1,RED_BLOCK>>>(d_ri,d_ro);
        bench_reduction_scrubbed<<<1,RED_BLOCK>>>(d_ri,d_ro);
        bench_gemm_baseline<<<1,GEMM_TILE*GEMM_TILE>>>(d_A,d_B,d_C);
        bench_gemm_scrubbed<<<1,GEMM_TILE*GEMM_TILE>>>(d_A,d_B,d_C);
        bench_attn_baseline<<<1,ATTN_BLOCK>>>(d_q,d_k,d_ao);
        bench_attn_scrubbed<<<1,ATTN_BLOCK>>>(d_q,d_k,d_ao);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t ev_s, ev_m, ev_e;
    CUDA_CHECK(cudaEventCreate(&ev_s));
    CUDA_CHECK(cudaEventCreate(&ev_m));
    CUDA_CHECK(cudaEventCreate(&ev_e));

    float base_us[N_WARMUP+N_REPS], scrub_us[N_WARMUP+N_REPS], ovhd_us[N_REPS];
    float iso_base_us[N_ISO_WARMUP+N_ISO_REPS];
    float iso_scrub_us[N_ISO_WARMUP+N_ISO_REPS];
    float iso_ovhd_us[N_ISO_REPS];
    float mb, ms;

    // ── REDUCTION interleaved ─────────────────────────────────────────────────
    fprintf(stderr,"[M7] reduction interleaved (%d+%d reps × %d) ...\n",
            N_WARMUP,N_REPS,N_BATCH);
    for(int r=0;r<N_WARMUP+N_REPS;r++){
        CUDA_CHECK(cudaEventRecord(ev_s));
        for(int i=0;i<N_BATCH;i++) bench_reduction_baseline<<<1,RED_BLOCK>>>(d_ri,d_ro);
        CUDA_CHECK(cudaEventRecord(ev_m));
        for(int i=0;i<N_BATCH;i++) bench_reduction_scrubbed<<<1,RED_BLOCK>>>(d_ri,d_ro);
        CUDA_CHECK(cudaEventRecord(ev_e));
        CUDA_CHECK(cudaEventSynchronize(ev_e));
        CUDA_CHECK(cudaEventElapsedTime(&mb,ev_s,ev_m));
        CUDA_CHECK(cudaEventElapsedTime(&ms,ev_m,ev_e));
        base_us[r]=mb*1000.f/N_BATCH;
        scrub_us[r]=ms*1000.f/N_BATCH;
    }
    for(int r=0;r<N_REPS;r++)
        ovhd_us[r]=scrub_us[N_WARMUP+r]-base_us[N_WARMUP+r];
    Stats rb=compute_stats(base_us+N_WARMUP,N_REPS);
    Stats rs=compute_stats(scrub_us+N_WARMUP,N_REPS);
    Stats ro=compute_stats(ovhd_us,N_REPS);
    int red_below=(fabsf(ro.med)<ro.std);

    int red_occ_n; float red_occ;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &red_occ_n,bench_reduction_baseline,RED_BLOCK,RED_N*sizeof(float)));
    red_occ=(float)(red_occ_n*RED_BLOCK)/prop.maxThreadsPerMultiProcessor;

    // ── GEMM interleaved ──────────────────────────────────────────────────────
    fprintf(stderr,"[M7] gemm interleaved (%d+%d reps × %d) ...\n",
            N_WARMUP,N_REPS,N_BATCH);
    for(int r=0;r<N_WARMUP+N_REPS;r++){
        CUDA_CHECK(cudaEventRecord(ev_s));
        for(int i=0;i<N_BATCH;i++) bench_gemm_baseline<<<1,GEMM_TILE*GEMM_TILE>>>(d_A,d_B,d_C);
        CUDA_CHECK(cudaEventRecord(ev_m));
        for(int i=0;i<N_BATCH;i++) bench_gemm_scrubbed<<<1,GEMM_TILE*GEMM_TILE>>>(d_A,d_B,d_C);
        CUDA_CHECK(cudaEventRecord(ev_e));
        CUDA_CHECK(cudaEventSynchronize(ev_e));
        CUDA_CHECK(cudaEventElapsedTime(&mb,ev_s,ev_m));
        CUDA_CHECK(cudaEventElapsedTime(&ms,ev_m,ev_e));
        base_us[r]=mb*1000.f/N_BATCH;
        scrub_us[r]=ms*1000.f/N_BATCH;
    }
    for(int r=0;r<N_REPS;r++)
        ovhd_us[r]=scrub_us[N_WARMUP+r]-base_us[N_WARMUP+r];
    Stats gb=compute_stats(base_us+N_WARMUP,N_REPS);
    Stats gs=compute_stats(scrub_us+N_WARMUP,N_REPS);
    Stats go=compute_stats(ovhd_us,N_REPS);
    int gemm_below=(fabsf(go.med)<go.std);

    int gemm_occ_n; float gemm_occ;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &gemm_occ_n,bench_gemm_baseline,GEMM_TILE*GEMM_TILE,
        2*GEMM_TILE*GEMM_TILE*sizeof(float)));
    gemm_occ=(float)(gemm_occ_n*GEMM_TILE*GEMM_TILE)/prop.maxThreadsPerMultiProcessor;

    // ── ATTENTION interleaved ─────────────────────────────────────────────────
    int attn_smem=(ATTN_SEQ*ATTN_HEAD+ATTN_SEQ)*(int)sizeof(float);
    fprintf(stderr,"[M7] attention interleaved (%d+%d reps × %d) ...\n",
            N_WARMUP,N_REPS,N_BATCH);
    for(int r=0;r<N_WARMUP+N_REPS;r++){
        CUDA_CHECK(cudaEventRecord(ev_s));
        for(int i=0;i<N_BATCH;i++) bench_attn_baseline<<<1,ATTN_BLOCK>>>(d_q,d_k,d_ao);
        CUDA_CHECK(cudaEventRecord(ev_m));
        for(int i=0;i<N_BATCH;i++) bench_attn_scrubbed<<<1,ATTN_BLOCK>>>(d_q,d_k,d_ao);
        CUDA_CHECK(cudaEventRecord(ev_e));
        CUDA_CHECK(cudaEventSynchronize(ev_e));
        CUDA_CHECK(cudaEventElapsedTime(&mb,ev_s,ev_m));
        CUDA_CHECK(cudaEventElapsedTime(&ms,ev_m,ev_e));
        base_us[r]=mb*1000.f/N_BATCH;
        scrub_us[r]=ms*1000.f/N_BATCH;
    }
    for(int r=0;r<N_REPS;r++)
        ovhd_us[r]=scrub_us[N_WARMUP+r]-base_us[N_WARMUP+r];
    Stats ab =compute_stats(base_us+N_WARMUP,N_REPS);
    Stats as_=compute_stats(scrub_us+N_WARMUP,N_REPS);
    Stats ao =compute_stats(ovhd_us,N_REPS);
    int attn_below=(fabsf(ao.med)<ao.std);

    int attn_occ_n; float attn_occ;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &attn_occ_n,bench_attn_baseline,ATTN_BLOCK,attn_smem));
    attn_occ=(float)(attn_occ_n*ATTN_BLOCK)/prop.maxThreadsPerMultiProcessor;

    // ── REDUCTION scrub-isolation ─────────────────────────────────────────────
    fprintf(stderr,"[M7] reduction scrub-isolation (%d+%d reps × %d) ...\n",
            N_ISO_WARMUP,N_ISO_REPS,N_ISO_BATCH);
    for(int r=0;r<N_ISO_WARMUP+N_ISO_REPS;r++){
        CUDA_CHECK(cudaEventRecord(ev_s));
        for(int i=0;i<N_ISO_BATCH;i++) bench_red_empty<<<1,RED_BLOCK>>>();
        CUDA_CHECK(cudaEventRecord(ev_m));
        for(int i=0;i<N_ISO_BATCH;i++) bench_red_scrub_only<<<1,RED_BLOCK>>>();
        CUDA_CHECK(cudaEventRecord(ev_e));
        CUDA_CHECK(cudaEventSynchronize(ev_e));
        CUDA_CHECK(cudaEventElapsedTime(&mb,ev_s,ev_m));
        CUDA_CHECK(cudaEventElapsedTime(&ms,ev_m,ev_e));
        iso_base_us[r]=mb*1000.f/N_ISO_BATCH;
        iso_scrub_us[r]=ms*1000.f/N_ISO_BATCH;
    }
    for(int r=0;r<N_ISO_REPS;r++)
        iso_ovhd_us[r]=iso_scrub_us[N_ISO_WARMUP+r]-iso_base_us[N_ISO_WARMUP+r];
    Stats red_iso     =compute_stats(iso_ovhd_us,N_ISO_REPS);
    Stats red_iso_base=compute_stats(iso_base_us+N_ISO_WARMUP,N_ISO_REPS);
    Stats red_iso_sc  =compute_stats(iso_scrub_us+N_ISO_WARMUP,N_ISO_REPS);

    // ── ATTENTION scrub-isolation ─────────────────────────────────────────────
    fprintf(stderr,"[M7] attention scrub-isolation (%d+%d reps × %d) ...\n",
            N_ISO_WARMUP,N_ISO_REPS,N_ISO_BATCH);
    for(int r=0;r<N_ISO_WARMUP+N_ISO_REPS;r++){
        CUDA_CHECK(cudaEventRecord(ev_s));
        for(int i=0;i<N_ISO_BATCH;i++) bench_attn_empty<<<1,ATTN_BLOCK>>>();
        CUDA_CHECK(cudaEventRecord(ev_m));
        for(int i=0;i<N_ISO_BATCH;i++) bench_attn_scrub_only<<<1,ATTN_BLOCK>>>();
        CUDA_CHECK(cudaEventRecord(ev_e));
        CUDA_CHECK(cudaEventSynchronize(ev_e));
        CUDA_CHECK(cudaEventElapsedTime(&mb,ev_s,ev_m));
        CUDA_CHECK(cudaEventElapsedTime(&ms,ev_m,ev_e));
        iso_base_us[r]=mb*1000.f/N_ISO_BATCH;
        iso_scrub_us[r]=ms*1000.f/N_ISO_BATCH;
    }
    for(int r=0;r<N_ISO_REPS;r++)
        iso_ovhd_us[r]=iso_scrub_us[N_ISO_WARMUP+r]-iso_base_us[N_ISO_WARMUP+r];
    Stats attn_iso     =compute_stats(iso_ovhd_us,N_ISO_REPS);
    Stats attn_iso_base=compute_stats(iso_base_us+N_ISO_WARMUP,N_ISO_REPS);
    Stats attn_iso_sc  =compute_stats(iso_scrub_us+N_ISO_WARMUP,N_ISO_REPS);

    // ── JSON output ───────────────────────────────────────────────────────────
    char red_occ_note[256], gemm_occ_note[256], attn_occ_note[256];
    snprintf(red_occ_note,sizeof(red_occ_note),
             "%.1f%% theoretical occupancy (sm_%d%d)",
             red_occ*100,prop.major,prop.minor);
    snprintf(gemm_occ_note,sizeof(gemm_occ_note),
             "%.1f%% theoretical occupancy (sm_%d%d)",
             gemm_occ*100,prop.major,prop.minor);
    snprintf(attn_occ_note,sizeof(attn_occ_note),
             "%.1f%% theoretical occupancy on sm_%d%d -- shared footprint "
             "(%d KB) limits to %d active blocks/SM (max %d threads/SM)",
             attn_occ*100,prop.major,prop.minor,
             attn_smem/1024,attn_occ_n,prop.maxThreadsPerMultiProcessor);

    printf("{\n");
    printf("  \"experiment\": \"Module 7 -- LMHC Benchmark Suite\",\n");
    printf("  \"hardware\": \"%s (sm_%d%d)\",\n",prop.name,prop.major,prop.minor);
    printf("  \"n_global_warmup\": %d,\n",N_GLOBAL_WARMUP);
    printf("  \"n_warmup_reps\": %d, \"n_reps\": %d, \"n_batch\": %d,\n",
           N_WARMUP,N_REPS,N_BATCH);
    printf("  \"n_iso_warmup\": %d, \"n_iso_reps\": %d, \"n_iso_batch\": %d,\n",
           N_ISO_WARMUP,N_ISO_REPS,N_ISO_BATCH);
    printf("  \"timing_method\": \"interleaved: %d baseline + %d scrubbed per rep (3 events, same stream); overhead = per-rep (scrub_half - base_half); median of %d reps\",\n",
           N_BATCH,N_BATCH,N_REPS);
    printf("  \"iso_method\": \"empty kernel (same smem) vs scrub-only kernel; %d reps x %d launches\",\n",
           N_ISO_REPS,N_ISO_BATCH);
    printf("  \"kernels\": [\n");

    Stats dummy={0.f,0.f};
    print_kernel_json(
        "reduction",
        "tree reduction (memory-bound), RED_N=4096 floats",
        RED_N*4.f/1024.f, RED_N, RED_BLOCK,
        rb, rs, ro, red_below,
        red_iso, red_iso_base, red_iso_sc, 1,
        red_occ, red_occ_note, 0);

    print_kernel_json(
        "gemm",
        "16x16 tiled GEMM (compute-bound), TILE=16 K=256",
        2*GEMM_TILE*GEMM_TILE*4.f/1024.f, GEMM_TILE*GEMM_TILE, GEMM_TILE*GEMM_TILE,
        gb, gs, go, gemm_below,
        dummy, dummy, dummy, 0,
        gemm_occ, gemm_occ_note, 0);

    print_kernel_json(
        "attention",
        "attention-style (memory-heavy), SEQ=64 HEAD=32",
        attn_smem/1024.f, ATTN_SEQ*ATTN_HEAD, ATTN_BLOCK,
        ab, as_, ao, attn_below,
        attn_iso, attn_iso_base, attn_iso_sc, 1,
        attn_occ, attn_occ_note, 1);

    printf("  ]\n}\n");

    cudaFree(d_ri); cudaFree(d_ro);
    cudaFree(d_A);  cudaFree(d_B); cudaFree(d_C);
    cudaFree(d_q);  cudaFree(d_k); cudaFree(d_ao);
    cudaEventDestroy(ev_s); cudaEventDestroy(ev_m); cudaEventDestroy(ev_e);
    return 0;
}
