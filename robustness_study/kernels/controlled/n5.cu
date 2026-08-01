#include <stdint.h>
#include <cuda_runtime.h>

// N5 -- stores, fence, barrier present but in INTENTIONALLY INVALID order.
// DO NOT EXECUTE -- compiled/disassembled only.
__launch_bounds__(32, 1)
__global__ void kernel_n5(uint32_t *output)
{
    int tid = threadIdx.x;
    __shared__ uint32_t sbuf[1024];
    for (int i = tid; i < 1024; i += 32)
        sbuf[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;
    __syncthreads();
    if (tid == 0) *output = sbuf[0] ^ sbuf[1023];
    __syncthreads();
    __syncthreads();
    __threadfence_block();
    volatile uint32_t *v = sbuf;
    for (int i = tid; i < 1024; i += 32)
        v[i] = 0u;
}

int main() {
    uint32_t *d_out;
    cudaMalloc(&d_out, sizeof(uint32_t));
    kernel_n5<<<1, 32>>>(d_out);
    cudaDeviceSynchronize();
    cudaFree(d_out);
    return 0;
}
