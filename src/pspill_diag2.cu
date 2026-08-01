// pspill_diag2.cu — diagnostic only: prints the generic-space address of the
// local array's first element on two consecutive launches of the SAME kernel,
// to determine whether CUDA local-memory backing store address is stable
// across launches (ruling in/out driver-side re-zeroing vs. address rotation
// as the explanation for the EI=0.0 result in pspill_diag.cu).

#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define LOCAL_WORDS 256
#define CUDA_CHECK(call) \
    do { cudaError_t _e=(call); \
         if(_e!=cudaSuccess){ fprintf(stderr,"CUDA error %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} \
    } while(0)

__global__ void probe_addr(unsigned long long *addr_out, uint32_t *val_out, int write_pattern){
    uint32_t local_arr[LOCAL_WORDS];
    if(write_pattern){
        #pragma unroll 1
        for(int i=0;i<LOCAL_WORDS;i++) local_arr[i]=0xCAFEBABEu;
    }
    if(threadIdx.x==0){
        *addr_out = (unsigned long long)(uintptr_t)&local_arr[0];
        *val_out  = local_arr[0];
    }
}

int main(void){
    unsigned long long *d_addr; uint32_t *d_val;
    CUDA_CHECK(cudaMalloc(&d_addr,sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_val,sizeof(uint32_t)));

    unsigned long long a1,a2; uint32_t v1,v2;
    probe_addr<<<1,32>>>(d_addr,d_val,1);   // launch 1: write pattern
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&a1,d_addr,8,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&v1,d_val,4,cudaMemcpyDeviceToHost));

    probe_addr<<<1,32>>>(d_addr,d_val,0);   // launch 2: read-only, no write
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(&a2,d_addr,8,cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&v2,d_val,4,cudaMemcpyDeviceToHost));

    printf("launch1 addr=0x%llx val=0x%08x\n",a1,v1);
    printf("launch2 addr=0x%llx val=0x%08x\n",a2,v2);
    printf("same_address=%s  residue_present=%s\n",
           (a1==a2)?"YES":"NO", (v2==0xCAFEBABEu)?"YES":"NO");
    return 0;
}
