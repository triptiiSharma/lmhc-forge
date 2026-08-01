// leakage_demo_pstage.cu — Phase VI.B: P-stage residue recovery
//
// Methodologically parallel to src/leakage_demo.cu (Module 6, P-reuse), but
// targets a DIFFERENT physical residue path: a writer kernel that explicitly
// stages a value shared->global as ordinary dataflow (mirroring how tiled
// GEMM/attention stage operands through a global-backed intermediate), then
// exits and frees that buffer. A scavenger allocates a same-size buffer
// afterward and reads it BEFORE writing, testing whether the CUDA device
// allocator (cudaMalloc/cudaFree) hands back the same physical/virtual
// backing store without zeroing it -- the P-stage mechanism described in the
// RP (Section I, "P-stage") and distinct from both P-reuse (on-chip SRAM,
// SM-residency-gated) and P-spill (per-thread local memory; see
// results/leakage_demo_pspill.json for that path's negative finding on this
// hardware/driver -- P-stage is a different allocator-level mechanism and is
// not assumed to behave the same way).
//
// MANDATORY FRAMING (docs/SPEC.md / docs/RESULTS.md, applies here identically):
//   "Residue is fully observable when the subsequent kernel is co-resident on
//    the same SM -- co-residency is forced in this experiment to demonstrate the
//    window. This is in-process cross-kernel shared-memory persistence, NOT
//    cross-tenant LeftoverLocals on NVIDIA."
//   For P-stage the relevant "co-residency" is allocator reuse (the scavenger's
//   cudaMalloc returning the same block the writer just cudaFree'd), gated and
//   reported explicitly below (same_alloc), analogous to the same-SM gate in
//   leakage_demo.cu.
//
// Phase A — writer WITHOUT scrub: stages PATTERN shared->global, frees the
//           buffer unscrubbed; scavenger's next cudaMalloc of the same size
//           is checked for pointer reuse and read before writing.
// Phase B — writer WITH LMHC hardened scrub of the global staging buffer
//           before exit (and before free).
// Control  — scavenger allocates and reads first, before any writer runs.
//
// Output: JSON to stdout -> results/leakage_demo_pstage.json

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define TILE_WORDS  4096          // 4096 x uint32 = 16 KB (matches leakage_demo.cu
                                   // buffer size, models a GEMM/attention tile)
#define NTRIALS      200
#define BLOCK_DIM     32
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

// ── Writer A: shared->global staging of PATTERN, exits WITHOUT scrub ────────
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelA_pstage_unscrubbed(uint32_t *g_staging, uint32_t *smid_out){
    __shared__ uint32_t tile[TILE_WORDS];
    int tid = threadIdx.x;
    if(tid==0) *smid_out = get_smid();
    // Phase 1: compute/fill the shared tile (models a GEMM/attention operand)
    for(int i=tid;i<TILE_WORDS;i+=BLOCK_DIM) tile[i]=PATTERN;
    __syncthreads();
    // Phase 2: explicit shared->global staging (ordinary dataflow)
    for(int i=tid;i<TILE_WORDS;i+=BLOCK_DIM) g_staging[i]=tile[i];
    __syncthreads();
    // <- EXIT WITHOUT SCRUBBING g_staging -- secret persists in the global
    //    allocation after this kernel exits and the buffer is freed.
}

// ── Writer B: same staging, exits WITH LMHC hardened scrub of g_staging ─────
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelA_pstage_scrubbed(uint32_t *g_staging, uint32_t *smid_out){
    __shared__ uint32_t tile[TILE_WORDS];
    int tid = threadIdx.x;
    if(tid==0) *smid_out = get_smid();
    for(int i=tid;i<TILE_WORDS;i+=BLOCK_DIM) tile[i]=PATTERN;
    __syncthreads();
    for(int i=tid;i<TILE_WORDS;i+=BLOCK_DIM) g_staging[i]=tile[i];
    __syncthreads();
    // ── LMHC hardened scrub (Algorithm 1, global-backed-region variant) ─────
    volatile uint32_t *vg = g_staging;
    for(int i=tid;i<TILE_WORDS;i+=BLOCK_DIM) vg[i]=0u;
    __threadfence();       // device-scope: staging buffer must be durably zero
                            // in DRAM before cudaFree() returns it to the pool.
    __syncthreads();
    // <- EXIT AFTER SCRUB
}

// ── Scavenger: reads the freed/reallocated global region BEFORE writing ─────
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelB_pstage_scavenger(uint32_t *g_region, int *nz_out,
                                          uint32_t *sample_out){
    int tid = threadIdx.x;
    int nz=0;
    for(int i=tid;i<TILE_WORDS;i+=BLOCK_DIM){
        uint32_t v = g_region[i];                 // === READ BEFORE WRITE ===
        if(v!=0u) nz++;
        if(tid==0 && i<BLOCK_DIM) sample_out[i]=v;
    }
    for(int off=16;off>0;off>>=1) nz+=__shfl_down_sync(0xFFFFFFFFu,nz,off);
    if(tid==0) atomicAdd(nz_out,nz);
}

enum WriterMode { NO_WRITER=0, UNSCRUBBED=1, SCRUBBED=2 };

static void run_phase(const char *phase_label, WriterMode mode,
                      uint32_t *d_smA, uint32_t *d_smB,
                      int *d_nz, uint32_t *d_sample,
                      int *same_alloc_out, int *diff_alloc_out,
                      long long *nz_same_out){
    int same_alloc=0, diff_alloc=0; long long nz_same=0;
    fprintf(stderr,"[Pstage] %s (%d trials) ...\n",phase_label,NTRIALS);
    for(int t=0;t<NTRIALS;t++){
        uintptr_t addr1=0;
        if(mode!=NO_WRITER){
            uint32_t *g1;
            CUDA_CHECK(cudaMalloc(&g1, TILE_WORDS*sizeof(uint32_t)));
            addr1 = (uintptr_t)g1;
            if(mode==UNSCRUBBED) kernelA_pstage_unscrubbed<<<1,BLOCK_DIM>>>(g1,d_smA);
            else                 kernelA_pstage_scrubbed<<<1,BLOCK_DIM>>>(g1,d_smA);
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaFree(g1));
        }
        uint32_t *g2;
        CUDA_CHECK(cudaMalloc(&g2, TILE_WORDS*sizeof(uint32_t)));
        uintptr_t addr2 = (uintptr_t)g2;
        CUDA_CHECK(cudaMemset(d_nz,0,sizeof(int)));
        kernelB_pstage_scavenger<<<1,BLOCK_DIM>>>(g2,d_nz,d_sample);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(g2));

        int hnz;
        CUDA_CHECK(cudaMemcpy(&hnz,d_nz,sizeof(int),cudaMemcpyDeviceToHost));

        if(mode==NO_WRITER){
            same_alloc++; nz_same+=(long long)hnz;   // control: no reuse gate needed
        } else if(addr1==addr2){
            same_alloc++; nz_same+=(long long)hnz;
        } else {
            diff_alloc++;
        }
    }
    *same_alloc_out=same_alloc; *diff_alloc_out=diff_alloc; *nz_same_out=nz_same;
    fprintf(stderr,"  same-alloc: %d/%d  diff-alloc: %d  nz_same: %lld/%lld\n",
            same_alloc,NTRIALS,diff_alloc,nz_same,(long long)same_alloc*TILE_WORDS);
}

int main(void){
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    fprintf(stderr,"[Pstage] %s (sm_%d%d)\n",prop.name,prop.major,prop.minor);

    uint32_t *d_smA,*d_smB,*d_sample; int *d_nz;
    CUDA_CHECK(cudaMalloc(&d_smA,4));
    CUDA_CHECK(cudaMalloc(&d_smB,4));
    CUDA_CHECK(cudaMalloc(&d_nz,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sample,BLOCK_DIM*4));

    int sameCtrl,diffCtrl,sameA,diffA,sameB,diffB;
    long long nzCtrl,nzAs,nzBs;
    run_phase("Control: scavenger only, no writer",NO_WRITER,
              d_smA,d_smB,d_nz,d_sample,&sameCtrl,&diffCtrl,&nzCtrl);
    run_phase("Phase A: writer WITHOUT scrub (EI_before)",UNSCRUBBED,
              d_smA,d_smB,d_nz,d_sample,&sameA,&diffA,&nzAs);
    run_phase("Phase B: writer WITH LMHC scrub (EI_after)",SCRUBBED,
              d_smA,d_smB,d_nz,d_sample,&sameB,&diffB,&nzBs);

    double ei_control = (sameCtrl>0) ? (double)nzCtrl/((long long)sameCtrl*TILE_WORDS) : 0.0;
    double ei_before  = (sameA>0)    ? (double)nzAs/((long long)sameA*TILE_WORDS)       : 0.0;
    double ei_after   = (sameB>0)    ? (double)nzBs/((long long)sameB*TILE_WORDS)        : 0.0;

    printf("{\n");
    printf("  \"experiment\": \"P-stage -- explicit shared->global staging-buffer residue recovery via cudaMalloc/cudaFree reuse\",\n");
    printf("  \"hardware\": \"%s (sm_%d%d)\",\n",prop.name,prop.major,prop.minor);
    printf("  \"framing\": \"Residue is fully observable when the scavenger's cudaMalloc call reuses the exact physical/virtual block the writer just cudaFree'd -- reuse is checked and gated explicitly per trial (same_alloc), analogous to the same-SM gate in leakage_demo.cu. This is in-process, single-context allocator reuse, NOT cross-tenant LeftoverLocals on NVIDIA and NOT a claim that the CUDA allocator is guaranteed to reuse freed blocks.\",\n");
    printf("  \"tile_words\": %d,\n",TILE_WORDS);
    printf("  \"block_dim\": %d,\n",BLOCK_DIM);
    printf("  \"pattern_sentinel\": \"0xCAFEBABE\",\n");
    printf("  \"trials_per_phase\": %d,\n",NTRIALS);
    printf("  \"control\": {\n");
    printf("    \"label\": \"scavenger only, writer never launched\",\n");
    printf("    \"same_alloc_trials\": %d,\n",sameCtrl);
    printf("    \"diff_alloc_trials\": %d,\n",diffCtrl);
    printf("    \"nonzero_words\": %lld,\n",(long long)nzCtrl);
    printf("    \"total_words\": %lld,\n",(long long)sameCtrl*TILE_WORDS);
    printf("    \"exposure_index\": %.6f\n",ei_control);
    printf("  },\n");
    printf("  \"phase_A\": {\n");
    printf("    \"label\": \"writer WITHOUT LMHC scrub\",\n");
    printf("    \"same_alloc_trials\": %d,\n",sameA);
    printf("    \"diff_alloc_trials\": %d,\n",diffA);
    printf("    \"nonzero_words\": %lld,\n",(long long)nzAs);
    printf("    \"total_words\": %lld,\n",(long long)sameA*TILE_WORDS);
    printf("    \"exposure_index\": %.6f\n",ei_before);
    printf("  },\n");
    printf("  \"phase_B\": {\n");
    printf("    \"label\": \"writer WITH LMHC hardened scrub\",\n");
    printf("    \"same_alloc_trials\": %d,\n",sameB);
    printf("    \"diff_alloc_trials\": %d,\n",diffB);
    printf("    \"nonzero_words\": %lld,\n",(long long)nzBs);
    printf("    \"total_words\": %lld,\n",(long long)sameB*TILE_WORDS);
    printf("    \"exposure_index\": %.6f\n",ei_after);
    printf("  },\n");
    printf("  \"exposure_index_control\": %.6f,\n",ei_control);
    printf("  \"exposure_index_before\": %.6f,\n",ei_before);
    printf("  \"exposure_index_after\":  %.6f,\n",ei_after);
    const char *result_text;
    if(sameA==0){
        result_text = "INCONCLUSIVE: the CUDA allocator never returned the same block to the scavenger across any trial (same_alloc_trials=0 in Phase A) -- the allocator-reuse precondition this experiment depends on did not hold on this run; no residue claim can be made either way.";
    } else if(ei_control==0.0 && ei_after==0.0 && ei_before>=0.99){
        result_text = "LMHC scrub reduces P-stage Exposure Index from ~1.0 to 0.0 on allocator-reuse trials -- global-backed staging-buffer residue window closed; control confirms no self-contamination.";
    } else if(ei_control==0.0 && ei_before==0.0 && ei_after==0.0){
        result_text = "NO RESIDUE OBSERVED on allocator-reuse trials (bounding result, not a defect) -- see P-spill's analogous finding (results/leakage_demo_pspill.json) for a parallel case on this hardware/driver.";
    } else {
        result_text = "WARNING: unexpected EI values -- check hardware/compilation.";
    }
    printf("  \"result\": \"%s\"\n", result_text);
    printf("}\n");

    cudaFree(d_smA); cudaFree(d_smB); cudaFree(d_nz); cudaFree(d_sample);
    return 0;
}
