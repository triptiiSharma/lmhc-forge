// unforced_coresidency.cu — RP §VI.E: unforced co-residency, natural scheduling
//
// SCOPE NOTE (per multi_gpu/README.md's own escape hatch: "if needed, create
// a separate exploratory path clearly labeled as non-controlled"): this file
// is intentionally SEPARATE from leakage_demo.cu, multi_gpu_repro.cu, and the
// Phase-0/Module-6 forced-co-residency results. It must never be averaged
// with, or presented as, the forced-co-residency measurement. It has its own
// results file (results/unforced_coresidency.json) and its own docs/RESULTS.md
// section.
//
// Design: SAME PROCESS, SAME CUDA CONTEXT, but NO __launch_bounds__ pinning
// and a multi-block grid, so the default scheduler is free to place blocks
// on whatever SMs it chooses (rather than forcing single-block/back-to-back
// same-SM placement as leakage_demo.cu does). We launch a writer kernel with
// NBLOCKS independent blocks (each with its own shared-memory buffer, filled
// with the sentinel), then a listener kernel with NBLOCKS independent blocks
// (each reading its OWN shared buffer before writing). For every listener
// block we check whether its %smid matches ANY writer block's %smid from the
// immediately preceding launch, and if so, whether it recovered non-zero
// words. This measures natural co-residency RATE, not a forced 100%/0% case.
//
// This does NOT test cross-process / cross-context residue (that requires
// either CUDA MPS or relying on the driver's normal per-context isolation,
// which is a separate, larger infrastructure question -- see docs/RESULTS.md for
// what was and was not attempted here).
//
// Output: JSON to stdout -> results/unforced_coresidency.json

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NWORDS_PER_BLOCK 16000      // 16000 x uint32 = 62.5 KB/block shared --
                                     // deliberately large (RTX 4050 has 100 KB/SM)
                                     // so at most ONE block is resident per SM at
                                     // a time; this caps occupancy so a partial
                                     // grid does NOT trivially touch every SM.
#define THREADS_PER_BLOCK    32      // one warp/block -- keeps the shuffle
                                     // reduction correct (single-warp, matches
                                     // leakage_demo.cu's proven-correct pattern).
#define WRITER_NBLOCKS        8      // fewer than SM count (20): writer does NOT
                                     // necessarily touch every SM, so listener
                                     // coincidence is a genuine, non-trivial rate.
#define LISTENER_NBLOCKS     20      // == SM count: one wave, spans all SMs.
#define NREPEATS             50
#define PATTERN  0xCAFEBABEu

#define CUDA_CHECK(call)                                                     \
    do { cudaError_t _e=(call);                                              \
         if(_e!=cudaSuccess){                                                \
             fprintf(stderr,"CUDA error %s:%d %s\n",                        \
                     __FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);}   \
    } while(0)

__device__ __forceinline__ uint32_t get_smid(){
    uint32_t id; asm volatile("mov.u32 %0, %%smid;" : "=r"(id)); return id;
}

// Dynamic shared memory (opt-in via cudaFuncSetAttribute in main()) --
// static __shared__ arrays cap at 48 KB/block on sm_89; we need ~62.5 KB/block
// to force at most one resident block per SM (100 KB/SM).
extern __shared__ uint32_t dyn_buf[];

// No __launch_bounds__ -- let the compiler/scheduler choose occupancy freely.
__global__ void writer_unforced(uint32_t *smid_out){
    uint32_t *buf = dyn_buf;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if(tid==0) smid_out[bid] = get_smid();
    for(int i=tid;i<NWORDS_PER_BLOCK;i+=THREADS_PER_BLOCK) buf[i]=PATTERN;
    __syncthreads();
    if(tid==0){ volatile uint32_t s=buf[NWORDS_PER_BLOCK-1]; (void)s; }
    // <- EXIT WITHOUT SCRUB
}

// Same as writer_unforced but exits WITH the LMHC hardened scrub applied.
__global__ void writer_unforced_scrubbed(uint32_t *smid_out){
    uint32_t *buf = dyn_buf;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if(tid==0) smid_out[bid] = get_smid();
    for(int i=tid;i<NWORDS_PER_BLOCK;i+=THREADS_PER_BLOCK) buf[i]=PATTERN;
    __syncthreads();
    if(tid==0){ volatile uint32_t s=buf[NWORDS_PER_BLOCK-1]; (void)s; }
    // ── LMHC hardened scrub (Module 2 pattern) ───────────────────────────────
    volatile uint32_t *vp = buf;
    for(int i=tid;i<NWORDS_PER_BLOCK;i+=THREADS_PER_BLOCK) vp[i]=0u;
    __threadfence_block();
    __syncthreads();
    // <- EXIT AFTER SCRUB
}

__global__ void listener_unforced(uint32_t *smid_out, int *nz_out){
    uint32_t *buf = dyn_buf;
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    if(tid==0) smid_out[bid] = get_smid();
    int nz=0;
    for(int i=tid;i<NWORDS_PER_BLOCK;i+=THREADS_PER_BLOCK){
        uint32_t v=buf[i];               // === READ BEFORE WRITE ===
        if(v!=0u) nz++;
    }
    for(int off=16;off>0;off>>=1) nz+=__shfl_down_sync(0xFFFFFFFFu,nz,off);
    if(tid==0) atomicAdd(&nz_out[bid],nz);
}

typedef void (*writer_kernel_t)(uint32_t*);

struct PhaseResult {
    long long listener_blocks_total, listener_blocks_coresident;
    long long coresident_nonzero, coresident_total_words;
    long long noncoresident_nonzero, noncoresident_total_words;
};

static PhaseResult run_phase(const char *label, writer_kernel_t writer_kernel,
                              size_t shmem_bytes,
                              uint32_t *d_smid_w, uint32_t *d_smid_l, int *d_nz,
                              uint32_t *h_smid_w, uint32_t *h_smid_l, int *h_nz){
    PhaseResult r{};
    fprintf(stderr,"[Unforced] %s (%d repeats) ...\n",label,NREPEATS);
    for(int rep=0; rep<NREPEATS; rep++){
        writer_kernel<<<WRITER_NBLOCKS,THREADS_PER_BLOCK,shmem_bytes>>>(d_smid_w);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemset(d_nz,0,LISTENER_NBLOCKS*sizeof(int)));
        listener_unforced<<<LISTENER_NBLOCKS,THREADS_PER_BLOCK,shmem_bytes>>>(d_smid_l,d_nz);
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(h_smid_w,d_smid_w,WRITER_NBLOCKS*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_smid_l,d_smid_l,LISTENER_NBLOCKS*sizeof(uint32_t),cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_nz,d_nz,LISTENER_NBLOCKS*sizeof(int),cudaMemcpyDeviceToHost));

        int writer_smid_present[256]={0};   // sm_89 has 20 SMs; 256 is a safe upper bound
        for(int b=0;b<WRITER_NBLOCKS;b++) writer_smid_present[h_smid_w[b]%256]=1;

        for(int b=0;b<LISTENER_NBLOCKS;b++){
            r.listener_blocks_total++;
            int coresident = writer_smid_present[h_smid_l[b]%256];
            if(coresident){
                r.listener_blocks_coresident++;
                r.coresident_nonzero += h_nz[b];
                r.coresident_total_words += NWORDS_PER_BLOCK;
            } else {
                r.noncoresident_nonzero += h_nz[b];
                r.noncoresident_total_words += NWORDS_PER_BLOCK;
            }
        }
    }
    double rate = (double)r.listener_blocks_coresident/(double)r.listener_blocks_total;
    double ei_c = (r.coresident_total_words>0)? (double)r.coresident_nonzero/(double)r.coresident_total_words : 0.0;
    fprintf(stderr,"  listener blocks: %lld total, %lld co-resident (%.4f rate); EI(co-resident)=%.6f\n",
            r.listener_blocks_total, r.listener_blocks_coresident, rate, ei_c);
    return r;
}

int main(void){
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    fprintf(stderr,"[Unforced] %s (sm_%d%d), %d SMs\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);

    size_t shmem_bytes = (size_t)NWORDS_PER_BLOCK*sizeof(uint32_t);
    CUDA_CHECK(cudaFuncSetAttribute(writer_unforced,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(writer_unforced_scrubbed,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem_bytes));
    CUDA_CHECK(cudaFuncSetAttribute(listener_unforced,
        cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shmem_bytes));
    fprintf(stderr,"[Unforced] dynamic shared mem/block: %zu bytes (%.1f KB) -- caps occupancy near 1 block/SM on a %d KB/SM device\n",
            shmem_bytes, shmem_bytes/1024.0, (int)(prop.sharedMemPerMultiprocessor/1024));

    uint32_t *d_smid_w,*d_smid_l; int *d_nz;
    CUDA_CHECK(cudaMalloc(&d_smid_w,WRITER_NBLOCKS*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_smid_l,LISTENER_NBLOCKS*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&d_nz,LISTENER_NBLOCKS*sizeof(int)));

    uint32_t *h_smid_w=(uint32_t*)malloc(WRITER_NBLOCKS*sizeof(uint32_t));
    uint32_t *h_smid_l=(uint32_t*)malloc(LISTENER_NBLOCKS*sizeof(uint32_t));
    int *h_nz=(int*)malloc(LISTENER_NBLOCKS*sizeof(int));

    PhaseResult before = run_phase("Phase A: writer WITHOUT LMHC scrub",writer_unforced,shmem_bytes,
                                    d_smid_w,d_smid_l,d_nz,h_smid_w,h_smid_l,h_nz);
    PhaseResult after  = run_phase("Phase B: writer WITH LMHC hardened scrub",writer_unforced_scrubbed,shmem_bytes,
                                    d_smid_w,d_smid_l,d_nz,h_smid_w,h_smid_l,h_nz);

    double rate_before = (double)before.listener_blocks_coresident/(double)before.listener_blocks_total;
    double rate_after  = (double)after.listener_blocks_coresident/(double)after.listener_blocks_total;
    double ei_before = (before.coresident_total_words>0)? (double)before.coresident_nonzero/(double)before.coresident_total_words : 0.0;
    double ei_after  = (after.coresident_total_words>0)?  (double)after.coresident_nonzero/(double)after.coresident_total_words  : 0.0;

    printf("{\n");
    printf("  \"experiment\": \"Unforced co-residency -- natural scheduler placement, same process/context, no __launch_bounds__ pinning\",\n");
    printf("  \"hardware\": \"%s (sm_%d%d), %d SMs\",\n",prop.name,prop.major,prop.minor,prop.multiProcessorCount);
    printf("  \"scope_note\": \"SAME OS process and CUDA context as leakage_demo.cu/Phase 0, but placement is NOT forced (no __launch_bounds__, multi-block grid). Does NOT test cross-process or cross-context residue (would require CUDA MPS or reliance on default per-context driver isolation -- not attempted here, see docs/RESULTS.md). This result must never be averaged with or substituted for the forced-co-residency Phase 0/Module 6 results.\",\n");
    printf("  \"words_per_block\": %d,\n",NWORDS_PER_BLOCK);
    printf("  \"threads_per_block\": %d,\n",THREADS_PER_BLOCK);
    printf("  \"writer_nblocks\": %d,\n",WRITER_NBLOCKS);
    printf("  \"listener_nblocks\": %d,\n",LISTENER_NBLOCKS);
    printf("  \"nrepeats\": %d,\n",NREPEATS);
    printf("  \"pattern_sentinel\": \"0xCAFEBABE\",\n");
    printf("  \"phase_A_unscrubbed\": {\n");
    printf("    \"listener_blocks_total\": %lld,\n",before.listener_blocks_total);
    printf("    \"listener_blocks_coresident_with_writer\": %lld,\n",before.listener_blocks_coresident);
    printf("    \"natural_coresidency_rate\": %.6f,\n",rate_before);
    printf("    \"exposure_index_coresident_blocks\": %.6f,\n",ei_before);
    printf("    \"coresident_nonzero_words\": %lld,\n",before.coresident_nonzero);
    printf("    \"coresident_total_words\": %lld\n",before.coresident_total_words);
    printf("  },\n");
    printf("  \"phase_B_lmhc_scrubbed\": {\n");
    printf("    \"listener_blocks_total\": %lld,\n",after.listener_blocks_total);
    printf("    \"listener_blocks_coresident_with_writer\": %lld,\n",after.listener_blocks_coresident);
    printf("    \"natural_coresidency_rate\": %.6f,\n",rate_after);
    printf("    \"exposure_index_coresident_blocks\": %.6f,\n",ei_after);
    printf("    \"coresident_nonzero_words\": %lld,\n",after.coresident_nonzero);
    printf("    \"coresident_total_words\": %lld\n",after.coresident_total_words);
    printf("  },\n");
    printf("  \"result\": \"Natural (unpinned, multi-block) co-residency rate = %.4f%% of listener blocks (both phases; placement is deterministic given fixed grid geometry, not scrub-dependent). Of co-resident blocks: EI = %.6f WITHOUT scrub, EI = %.6f WITH LMHC hardened scrub. This is a device- and occupancy-configuration-specific measurement (WRITER_NBLOCKS=%d, LISTENER_NBLOCKS=%d, %d SMs), not a claim about production scheduling frequency on any other hardware, grid size, or shared-memory footprint.\"\n",
           rate_before*100.0, ei_before, ei_after, WRITER_NBLOCKS, LISTENER_NBLOCKS, prop.multiProcessorCount);
    printf("}\n");

    free(h_smid_w); free(h_smid_l); free(h_nz);
    cudaFree(d_smid_w); cudaFree(d_smid_l); cudaFree(d_nz);
    return 0;
}
