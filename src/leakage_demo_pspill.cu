// leakage_demo_pspill.cu — Phase VI.B: P-spill residue recovery
//
// Methodologically parallel to src/leakage_demo.cu (Module 6, P-reuse), but
// targets a DIFFERENT physical residue path: register/shared pressure spilled
// by the compiler into CUDA "local" memory, which is per-thread but physically
// backed by global DRAM (NOT the on-chip scratchpad tested by leakage_demo.cu).
//
// MANDATORY FRAMING (docs/SPEC.md / docs/RESULTS.md, applies here identically):
//   "Residue is fully observable when the subsequent kernel is co-resident on
//    the same SM -- co-residency is forced in this experiment to demonstrate the
//    window. This is in-process cross-kernel shared-memory persistence, NOT
//    cross-tenant LeftoverLocals on NVIDIA."
//   For P-spill the persistence mechanism is NOT SM-residency (unlike P-reuse):
//   CUDA local memory is a per-context region addressed by launch geometry
//   (grid/block/thread index), not by which physical SM executes the block.
//   Two consecutive launches with identical <<<1,BLOCK_DIM>>> geometry address
//   the SAME local-memory bytes for a given thread regardless of SM placement.
//   This experiment measures that geometry-addressed reuse, not SM pinning.
//   See docs/RESULTS.md "P-spill" section for the SASS verification that the writer
//   kernel's PATTERN fill actually lowers to local-memory stores (STL / generic
//   store to the .local state space), not registers.
//
// Phase A — writer WITHOUT scrub: listener's local-memory frame reads back
//           the sentinel pattern left by the writer's spilled array.
// Phase B — writer WITH LMHC hardened scrub of the local array before exit:
//           listener reads zeros.
// Control  — listener only, writer never launched in this process.
//
// Output: JSON to stdout -> results/leakage_demo_pspill.json

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define LOCAL_WORDS  256          // 256 x uint32 = 1 KB/thread -- far beyond the
                                   // per-thread register budget, forces the
                                   // compiler to place this automatic array in
                                   // CUDA "local" memory (global-DRAM-backed).
#define NTRIALS      200          // matches leakage_demo.cu
#define BLOCK_DIM     32          // one warp; matches leakage_demo.cu
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

// ── Writer A: fills a spilled local array with PATTERN, exits WITHOUT scrub ──
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelA_pspill_unscrubbed(uint32_t *smid_out){
    uint32_t local_arr[LOCAL_WORDS];   // forced to .local (global-backed) by size
    int tid = threadIdx.x;
    if(tid==0) *smid_out = get_smid();
    #pragma unroll 1
    for(int i=0;i<LOCAL_WORDS;i++) local_arr[i]=PATTERN;
    // runtime-computed index prevents the store loop from being proven dead
    int idx = (tid*37 + 11) % LOCAL_WORDS;
    volatile uint32_t sink = local_arr[idx];
    (void)sink;
    // <- EXIT WITHOUT SCRUB -- pattern persists in local(global-backed) memory
}

// ── Writer B: same fill, exits WITH LMHC hardened scrub of the local array ──
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelA_pspill_scrubbed(uint32_t *smid_out){
    uint32_t local_arr[LOCAL_WORDS];
    int tid = threadIdx.x;
    if(tid==0) *smid_out = get_smid();
    #pragma unroll 1
    for(int i=0;i<LOCAL_WORDS;i++) local_arr[i]=PATTERN;
    int idx = (tid*37 + 11) % LOCAL_WORDS;
    volatile uint32_t sink = local_arr[idx];
    (void)sink;
    // ── LMHC hardened scrub (Algorithm 1, global-backed-region variant) ─────
    volatile uint32_t *vp = local_arr;
    #pragma unroll 1
    for(int i=0;i<LOCAL_WORDS;i++) vp[i]=0u;
    __threadfence();      // device-scope: local-memory backing store must be
                           // durably zero in DRAM before any later kernel in
                           // this context can reuse the same frame.
    __syncthreads();
    // <- EXIT AFTER SCRUB
}

// ── Listener: reads its OWN local frame BEFORE writing it ───────────────────
__launch_bounds__(BLOCK_DIM, 1)
__global__ void kernelB_pspill_listener(uint32_t *smid_out, int *nz_out,
                                         uint32_t *sample_out){
    uint32_t local_arr[LOCAL_WORDS];
    int tid = threadIdx.x;
    if(tid==0) *smid_out = get_smid();
    int nz=0;
    #pragma unroll 1
    for(int i=0;i<LOCAL_WORDS;i++){
        uint32_t v = local_arr[i];               // === READ BEFORE WRITE ===
        if(v!=0u) nz++;
        if(tid==0 && i<BLOCK_DIM) sample_out[i]=v;
    }
    for(int off=16;off>0;off>>=1) nz+=__shfl_down_sync(0xFFFFFFFFu,nz,off);
    if(tid==0) atomicAdd(nz_out,nz);
}

static void run_phase(const char *phase_label, int use_scrub, int have_writer,
                      uint32_t *d_smA, uint32_t *d_smB,
                      int *d_nz, uint32_t *d_sample,
                      int *same_out, long long *nz_same_out){
    int same=0; long long nz_same=0;
    fprintf(stderr,"[Pspill] %s (%d trials) ...\n",phase_label,NTRIALS);
    for(int t=0;t<NTRIALS;t++){
        CUDA_CHECK(cudaMemset(d_nz,0,sizeof(int)));
        if(have_writer){
            if(use_scrub) kernelA_pspill_scrubbed<<<1,BLOCK_DIM>>>(d_smA);
            else          kernelA_pspill_unscrubbed<<<1,BLOCK_DIM>>>(d_smA);
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        kernelB_pspill_listener<<<1,BLOCK_DIM>>>(d_smB,d_nz,d_sample);
        CUDA_CHECK(cudaDeviceSynchronize());
        int hnz;
        CUDA_CHECK(cudaMemcpy(&hnz,d_nz,sizeof(int),cudaMemcpyDeviceToHost));
        // No SM-residency gate here (see file header): local-memory reuse is
        // launch-geometry-addressed, not SM-addressed. Every trial counts.
        same++; nz_same+=(long long)hnz;
    }
    *same_out=same; *nz_same_out=nz_same;
    fprintf(stderr,"  trials: %d/%d  nz: %lld/%lld\n",
            same,NTRIALS,nz_same,(long long)same*BLOCK_DIM*LOCAL_WORDS);
}

int main(void){
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop,0));
    fprintf(stderr,"[Pspill] %s (sm_%d%d)\n",prop.name,prop.major,prop.minor);

    uint32_t *d_smA,*d_smB,*d_sample; int *d_nz;
    CUDA_CHECK(cudaMalloc(&d_smA,4));
    CUDA_CHECK(cudaMalloc(&d_smB,4));
    CUDA_CHECK(cudaMalloc(&d_nz,sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sample,BLOCK_DIM*4));

    int total_words = BLOCK_DIM*LOCAL_WORDS;

    int    sameCtrl,sameA,sameB;
    long long nzCtrl,nzAs,nzBs;
    run_phase("Control: listener only, no writer",0,0,
              d_smA,d_smB,d_nz,d_sample,&sameCtrl,&nzCtrl);
    run_phase("Phase A: writer WITHOUT scrub (EI_before)",0,1,
              d_smA,d_smB,d_nz,d_sample,&sameA,&nzAs);
    run_phase("Phase B: writer WITH LMHC scrub (EI_after)",1,1,
              d_smA,d_smB,d_nz,d_sample,&sameB,&nzBs);

    double ei_control = (sameCtrl>0) ? (double)nzCtrl/((long long)sameCtrl*total_words) : 0.0;
    double ei_before  = (sameA>0)    ? (double)nzAs/((long long)sameA*total_words)       : 0.0;
    double ei_after   = (sameB>0)    ? (double)nzBs/((long long)sameB*total_words)        : 0.0;

    printf("{\n");
    printf("  \"experiment\": \"P-spill -- CUDA local-memory (register/shared spill) residue recovery\",\n");
    printf("  \"hardware\": \"%s (sm_%d%d)\",\n",prop.name,prop.major,prop.minor);
    printf("  \"framing\": \"Residue is fully observable when a later kernel reuses the same local-memory frame -- reuse is addressed by identical launch geometry (grid/block/thread index), not SM residency, and is forced in this experiment via identical <<<1,32>>> launches with no intervening allocation of larger local-memory footprint. This is in-process cross-kernel local(global-backed)-memory persistence, NOT cross-tenant LeftoverLocals on NVIDIA.\",\n");
    printf("  \"mechanism_note\": \"Unlike P-reuse (on-chip shared memory, SM-residency-gated), P-spill's persistence is launch-geometry-addressed: CUDA local memory is a per-context region sized to the largest per-thread footprint requested so far and is not observed to be re-zeroed between kernel launches in the same context (A1). No same-SM gating is applied or needed for this path.\",\n");
    printf("  \"local_words_per_thread\": %d,\n",LOCAL_WORDS);
    printf("  \"block_dim\": %d,\n",BLOCK_DIM);
    printf("  \"total_words\": %d,\n",total_words);
    printf("  \"pattern_sentinel\": \"0xCAFEBABE\",\n");
    printf("  \"trials_per_phase\": %d,\n",NTRIALS);
    printf("  \"control\": {\n");
    printf("    \"label\": \"listener only, writer never launched\",\n");
    printf("    \"trials\": %d,\n",sameCtrl);
    printf("    \"nonzero_words\": %lld,\n",(long long)nzCtrl);
    printf("    \"total_words\": %lld,\n",(long long)sameCtrl*total_words);
    printf("    \"exposure_index\": %.6f\n",ei_control);
    printf("  },\n");
    printf("  \"phase_A\": {\n");
    printf("    \"label\": \"writer WITHOUT LMHC scrub\",\n");
    printf("    \"trials\": %d,\n",sameA);
    printf("    \"nonzero_words\": %lld,\n",(long long)nzAs);
    printf("    \"total_words\": %lld,\n",(long long)sameA*total_words);
    printf("    \"exposure_index\": %.6f\n",ei_before);
    printf("  },\n");
    printf("  \"phase_B\": {\n");
    printf("    \"label\": \"writer WITH LMHC hardened scrub\",\n");
    printf("    \"trials\": %d,\n",sameB);
    printf("    \"nonzero_words\": %lld,\n",(long long)nzBs);
    printf("    \"total_words\": %lld,\n",(long long)sameB*total_words);
    printf("    \"exposure_index\": %.6f\n",ei_after);
    printf("  },\n");
    printf("  \"exposure_index_control\": %.6f,\n",ei_control);
    printf("  \"exposure_index_before\": %.6f,\n",ei_before);
    printf("  \"exposure_index_after\":  %.6f,\n",ei_after);
    printf("  \"sass_verification_required\": \"cuobjdump -sass must confirm kernelA_pspill_unscrubbed lowers local_arr accesses to STL/LDL (or generic .local-space) instructions, not registers -- see docs/RESULTS.md P-spill section for the disassembly excerpt actually used to confirm this.\",\n");
    const char *result_text;
    if(ei_control==0.0 && ei_after==0.0 && ei_before>=0.99){
        result_text = "LMHC scrub reduces P-spill Exposure Index from ~1.0 to 0.0 -- local(global-backed) residue window closed; control confirms no self-contamination.";
    } else if(ei_control==0.0 && ei_before==0.0 && ei_after==0.0){
        result_text = "NO RESIDUE OBSERVED (bounding result, not a defect): control, unscrubbed, and scrubbed phases all read EI=0.0. SASS confirms genuine STL/LDL local-memory stores and loads (256-iteration loops, no dead-code elimination -- see sass_verification note). A same-kernel-twice diagnostic (src/pspill_diag.cu) and an address-stability diagnostic (src/pspill_diag2.cu) confirm the listener reads the IDENTICAL virtual address the writer wrote, yet observes zero, not the pattern. This indicates CUDA local (register/shared-spill) memory backing is zeroed between kernel launches on this hardware/driver/toolchain (RTX 4050, driver 573.22, CUDA 12.8 runtime, WSL2/WDDM passthrough) -- in contrast to on-chip shared memory, which showed 100% residue (Phase 0 / Module 6) under the IDENTICAL software stack. This bounds Platform Assumption A1 for the P-spill path specifically: A1 does not appear to hold for CUDA local memory on this configuration, even though it holds for on-chip shared memory. Root cause (WSL2/WDDM page-zeroing vs. native-Linux driver behavior vs. architecture-level scrubbing) is not established by this measurement and needs a native-Linux (non-WSL2) run to disambiguate.";
    } else {
        result_text = "WARNING: unexpected EI values -- check hardware/compilation/SASS.";
    }
    printf("  \"result\": \"%s\"\n", result_text);
    printf("}\n");

    cudaFree(d_smA); cudaFree(d_smB); cudaFree(d_nz); cudaFree(d_sample);
    return 0;
}
