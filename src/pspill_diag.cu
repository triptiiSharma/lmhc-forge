// pspill_diag.cu — diagnostic only (not a deliverable): isolates whether
// CUDA local-memory persistence requires the SAME __global__ function to be
// launched twice, vs. persisting across two DIFFERENT kernel functions with
// identical local-array size (leakage_demo_pspill.cu tests the latter, which
// came back EI=0.0 across all phases including the unscrubbed one).
//
// Single kernel, called twice with a mode flag: mode=0 writes PATTERN and
// exits (no scrub); mode=1 reads its own local frame BEFORE writing anything
// and reports nonzero-word count. Because it is the SAME compiled function
// both times, this isolates "same-kernel local-memory reuse" from
// "cross-kernel local-memory reuse".

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define LOCAL_WORDS 256
#define BLOCK_DIM 32
#define PATTERN 0xCAFEBABEu

#define CUDA_CHECK(call) \
    do { cudaError_t _e=(call); \
         if(_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} \
    } while(0)

__launch_bounds__(BLOCK_DIM,1)
__global__ void probe(int mode, int *nz_out){
    uint32_t local_arr[LOCAL_WORDS];
    int tid = threadIdx.x;
    if(mode==1){
        int nz=0;
        #pragma unroll 1
        for(int i=0;i<LOCAL_WORDS;i++){
            uint32_t v = local_arr[i];   // read BEFORE write
            if(v!=0u) nz++;
        }
        for(int off=16;off>0;off>>=1) nz+=__shfl_down_sync(0xFFFFFFFFu,nz,off);
        if(tid==0) atomicAdd(nz_out,nz);
    } else {
        #pragma unroll 1
        for(int i=0;i<LOCAL_WORDS;i++) local_arr[i]=PATTERN;
        int idx=(tid*37+11)%LOCAL_WORDS;
        volatile uint32_t sink=local_arr[idx]; (void)sink;
    }
}

int main(void){
    int *d_nz; CUDA_CHECK(cudaMalloc(&d_nz,sizeof(int)));
    int N=50, total_same=0; long long total_nz=0;
    for(int t=0;t<N;t++){
        CUDA_CHECK(cudaMemset(d_nz,0,sizeof(int)));
        probe<<<1,BLOCK_DIM>>>(0,d_nz);           // mode=0: write PATTERN, no scrub
        CUDA_CHECK(cudaDeviceSynchronize());
        probe<<<1,BLOCK_DIM>>>(1,d_nz);           // mode=1: SAME kernel, read-before-write
        CUDA_CHECK(cudaDeviceSynchronize());
        int hnz; CUDA_CHECK(cudaMemcpy(&hnz,d_nz,sizeof(int),cudaMemcpyDeviceToHost));
        total_nz+=hnz; total_same++;
    }
    double ei = (double)total_nz/((double)total_same*BLOCK_DIM*LOCAL_WORDS);
    printf("same-kernel-twice diagnostic: trials=%d nz=%lld/%lld EI=%.6f\n",
           total_same, total_nz, (long long)total_same*BLOCK_DIM*LOCAL_WORDS, ei);
    cudaFree(d_nz);
    return 0;
}
