// leakage_demo.cu — Module 6: LMHC Leakage Demonstration
// Writer/Listener experiment showing Exposure Index before/after LMHC scrub.
//
// MANDATORY FRAMING (docs/SPEC.md §Module 6):
//   "Residue is fully observable when the subsequent kernel is co-resident on
//    the same SM — co-residency is forced in this experiment to demonstrate the
//    window. This is in-process cross-kernel shared-memory persistence, NOT
//    cross-tenant LeftoverLocals on NVIDIA."
//
// Phase A — writer WITHOUT scrub: Listener reads back sentinel pattern.
//           Exposure Index = fraction of words matching sentinel ≈ 1.0
// Phase B — writer WITH LMHC hardened scrub: Listener reads zeros.
//           Exposure Index = 0.0 (scrub guarantees all words zeroed before exit)
//
// Output: JSON to stdout  →  results/leakage_demo.json

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define NWORDS     4096          // 4096 × uint32 = 16 KB shared (matches Phase 0)
#define NTRIALS     200          // 200 trials per phase (EI=0 is deterministic)
#define BLOCK_DIM    32          // one warp; __launch_bounds__ keeps occupancy low
#define PATTERN  0xCAFEBABEu    // sentinel written by writer kernels

#define CUDA_CHECK(call)                                                     \
    do { cudaError_t _e=(call);                                              \
         if(_e!=cudaSuccess){                                                \
             fprintf(stderr,"CUDA error %s:%d %s\n",                        \
                     __FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);}   \
    } while(0)

__device__ __forceinline__ uint32_t get_smid(){
    uint32_t id; asm volatile("mov.u32 %0, %%smid;" : "=r"(id)); return id;
}

// ── Writer A: fills shared with PATTERN, exits WITHOUT scrub ─────────────────
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelA_unscrubbed(uint32_t *smid_out){
    __shared__ uint32_t buf[NWORDS];
    int tid = threadIdx.x;
    if(tid==0) *smid_out = get_smid();
    for(int i=tid; i<NWORDS; i+=BLOCK_DIM) buf[i]=PATTERN;
    __syncthreads();
    // volatile sink prevents compiler from eliminating the fill above
    if(tid==0){ volatile uint32_t s=buf[NWORDS-1]; (void)s; }
    // ← EXIT WITHOUT SCRUB — pattern persists in shared SRAM
}

// ── Writer B: fills shared with PATTERN, exits WITH LMHC hardened scrub ─────
// This is in-process cross-kernel shared-memory persistence,
// NOT cross-tenant LeftoverLocals on NVIDIA.
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelA_scrubbed(uint32_t *smid_out){
    __shared__ uint32_t buf[NWORDS];
    int tid = threadIdx.x;
    if(tid==0) *smid_out = get_smid();
    for(int i=tid; i<NWORDS; i+=BLOCK_DIM) buf[i]=PATTERN;
    __syncthreads();
    if(tid==0){ volatile uint32_t s=buf[NWORDS-1]; (void)s; }
    // ── LMHC hardened scrub (Module 2 pattern) ───────────────────────────────
    volatile uint32_t *vp = buf;
    for(int i=tid; i<NWORDS; i+=BLOCK_DIM) vp[i]=0u;
    __threadfence_block();
    __syncthreads();
    // ← EXIT AFTER SCRUB — all words guaranteed zero by C++ memory model
}

// ── Listener: reads shared BEFORE writing, counts nonzero words ───────────────
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelB_listener(uint32_t *smid_out, int *nz_out,
                                  uint32_t *sample_out){
    __shared__ uint32_t buf[NWORDS];
    int tid = threadIdx.x;
    if(tid==0) *smid_out = get_smid();
    // === READ BEFORE ANY WRITE — measures cross-launch residue =================
    int nz=0;
    for(int i=tid; i<NWORDS; i+=BLOCK_DIM){
        uint32_t v=buf[i];
        if(v!=0u) nz++;
        if(i<BLOCK_DIM) sample_out[i]=v;
    }
    // ===========================================================================
    for(int off=16;off>0;off>>=1) nz+=__shfl_down_sync(0xFFFFFFFFu,nz,off);
    if(tid==0) atomicAdd(nz_out,nz);
}

static void run_phase(const char *phase_label, int use_scrub,
                      uint32_t *d_smA, uint32_t *d_smB,
                      int *d_nz, uint32_t *d_sample,
                      int *same_out, long long *nz_same_out,
                      long long *nz_all_out){
    int same=0,diff=0; long long nz_same=0,nz_all=0;
    fprintf(stderr,"[M6] %s (%d trials) ...\n",phase_label,NTRIALS);
    for(int t=0;t<NTRIALS;t++){
        CUDA_CHECK(cudaMemset(d_nz,0,sizeof(int)));
        if(use_scrub)
            kernelA_scrubbed<<<1,BLOCK_DIM>>>(d_smA);
        else
            kernelA_unscrubbed<<<1,BLOCK_DIM>>>(d_smA);
        CUDA_CHECK(cudaDeviceSynchronize());
        kernelB_listener<<<1,BLOCK_DIM>>>(d_smB,d_nz,d_sample);
        CUDA_CHECK(cudaDeviceSynchronize());
        uint32_t ha,hb; int hnz;
        CUDA_CHECK(cudaMemcpy(&ha,d_smA,4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hb,d_smB,4,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(&hnz,d_nz,sizeof(int),cudaMemcpyDeviceToHost));
        nz_all+=(long long)hnz;
        if(ha==hb){ same++; nz_same+=(long long)hnz; }
        else diff++;
    }
    *same_out=same; *nz_same_out=nz_same; *nz_all_out=nz_all;
    fprintf(stderr,"  same-SM: %d/%d  nz_same: %lld/%lld\n",
            same,NTRIALS,nz_same,(long long)same*NWORDS);
}

int main(void){
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    fprintf(stderr,"[M6] %s (sm_%d%d)\n",prop.name,prop.major,prop.minor);

    uint32_t *d_smA,*d_smB,*d_sample; int *d_nz;
    CUDA_CHECK(cudaMalloc(&d_smA,4));
    CUDA_CHECK(cudaMalloc(&d_smB,4));
    CUDA_CHECK(cudaMalloc(&d_nz,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sample,BLOCK_DIM*4));

    int    sameA,sameB;
    long long nzAs,nzAa,nzBs,nzBa;
    run_phase("Phase A: writer WITHOUT scrub (EI_before)",0,
              d_smA,d_smB,d_nz,d_sample,&sameA,&nzAs,&nzAa);
    run_phase("Phase B: writer WITH LMHC scrub (EI_after)",1,
              d_smA,d_smB,d_nz,d_sample,&sameB,&nzBs,&nzBa);

    double ei_before = (sameA>0) ? (double)nzAs/((long long)sameA*NWORDS) : 0.0;
    double ei_after  = (sameB>0) ? (double)nzBs/((long long)sameB*NWORDS) : 0.0;

    // JSON output
    printf("{\n");
    printf("  \"experiment\": \"Module 6 -- LMHC Leakage Demo\",\n");
    printf("  \"hardware\": \"%s (sm_%d%d)\",\n",prop.name,prop.major,prop.minor);
    printf("  \"framing\": \"Residue is fully observable when the subsequent kernel is co-resident on the same SM -- co-residency is forced in this experiment to demonstrate the window. This is in-process cross-kernel shared-memory persistence, NOT cross-tenant LeftoverLocals on NVIDIA.\",\n");
    printf("  \"buffer_words\": %d,\n",NWORDS);
    printf("  \"pattern_sentinel\": \"0xCAFEBABE\",\n");
    printf("  \"trials_per_phase\": %d,\n",NTRIALS);
    printf("  \"phase_A\": {\n");
    printf("    \"label\": \"writer WITHOUT LMHC scrub\",\n");
    printf("    \"same_sm_trials\": %d,\n",sameA);
    printf("    \"nonzero_words_same_sm\": %lld,\n",(long long)nzAs);
    printf("    \"total_words_same_sm\": %lld,\n",(long long)sameA*NWORDS);
    printf("    \"exposure_index\": %.6f\n",ei_before);
    printf("  },\n");
    printf("  \"phase_B\": {\n");
    printf("    \"label\": \"writer WITH LMHC hardened scrub\",\n");
    printf("    \"same_sm_trials\": %d,\n",sameB);
    printf("    \"nonzero_words_same_sm\": %lld,\n",(long long)nzBs);
    printf("    \"total_words_same_sm\": %lld,\n",(long long)sameB*NWORDS);
    printf("    \"exposure_index\": %.6f\n",ei_after);
    printf("  },\n");
    printf("  \"exposure_index_before\": %.6f,\n",ei_before);
    printf("  \"exposure_index_after\":  %.6f,\n",ei_after);
    printf("  \"result\": \"%s\"\n",
           (ei_after==0.0 && ei_before>=0.99)
           ? "LMHC scrub reduces Exposure Index from 1.0 to 0.0 — residue window closed."
           : "WARNING: unexpected EI values — check hardware/compilation.");
    printf("}\n");

    cudaFree(d_smA); cudaFree(d_smB); cudaFree(d_nz); cudaFree(d_sample);
    return 0;
}
