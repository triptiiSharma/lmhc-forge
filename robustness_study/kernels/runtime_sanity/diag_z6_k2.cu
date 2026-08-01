#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

__launch_bounds__(32, 1)
__global__ void diag_z6_k2(uint32_t *output, uint32_t *diag_out)
{
    int tid = threadIdx.x;
    __shared__ uint32_t sbuf[1024];
    for (int i = tid; i < 1024; i += 32)
        sbuf[i] = (uint32_t)(tid * 1234567u + (uint32_t)i) ^ 0xDEADBEEFu;
    __syncthreads();
    if (tid == 0) *output = sbuf[0] ^ sbuf[1023];
    __syncthreads();
    for (int i = tid; i < 1024; i += 32) {
        unsigned int saddr_sbuf = __cvta_generic_to_shared(&sbuf[i]);
        asm volatile("st.shared.u32 [%0], %1;" : : "r"(saddr_sbuf), "r"(0u) : "memory");
    }
    __threadfence_block();
    __syncthreads();
    for (int i = tid; i < 1024; i += 32)
        diag_out[i] = (uint32_t)sbuf[i];
}

int main() {
    uint32_t *d_out, *d_diag;
    cudaMalloc(&d_out, sizeof(uint32_t));
    cudaMalloc(&d_diag, 1024 * sizeof(uint32_t));
    for (int rep = 0; rep < 5; rep++) {
        diag_z6_k2<<<1, 32>>>(d_out, d_diag);
        cudaError_t launch_err = cudaGetLastError();
        cudaError_t sync_err = cudaDeviceSynchronize();
        uint32_t h_out;
        uint32_t h_diag[1024];
        cudaMemcpy(&h_out, d_out, sizeof(uint32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_diag, d_diag, 1024 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        int nonzero = 0;
        for (int i = 0; i < 1024; i++) if (h_diag[i] != 0u) nonzero++;
        printf("REP %d launch=%s sync=%s output=%u nonzero_words=%d\n", rep,
               cudaGetErrorString(launch_err), cudaGetErrorString(sync_err), h_out, nonzero);
    }
    cudaFree(d_out); cudaFree(d_diag);
    return 0;
}
