#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

__launch_bounds__(256, 1)
__global__ void diag_z6_k3(const float *input, float *output, int n, uint32_t *diag_out)
{
    int tid = threadIdx.x;
    __shared__ float sbuf[256];
    sbuf[tid] = (tid < n) ? input[blockIdx.x * 256 + tid] : 0.0f;
    __syncthreads();
    for (int stride = 128; stride > 0; stride >>= 1) {
        if (tid < stride) sbuf[tid] += sbuf[tid + stride];
        __syncthreads();
    }
    if (tid == 0) *output = sbuf[0];
    for (int i = tid; i < 256; i += 256) {
        unsigned int saddr_sbuf = __cvta_generic_to_shared(&sbuf[i]);
        asm volatile("st.shared.u32 [%0], %1;" : : "r"(saddr_sbuf), "r"(0u) : "memory");
    }
    __threadfence_block();
    __syncthreads();
    for (int i = tid; i < 256; i += 256)
        diag_out[i] = __float_as_uint(sbuf[i]);
}

int main() {
    float *d_in, *d_out; uint32_t *d_diag;
    float h_in[256];
    for (int i = 0; i < 256; i++) h_in[i] = (float)(i + 1);
    cudaMalloc(&d_in, 256 * sizeof(float));
    cudaMalloc(&d_out, sizeof(float));
    cudaMalloc(&d_diag, 256 * sizeof(uint32_t));
    cudaMemcpy(d_in, h_in, 256 * sizeof(float), cudaMemcpyHostToDevice);
    for (int rep = 0; rep < 5; rep++) {
        diag_z6_k3<<<1, 256>>>(d_in, d_out, 256, d_diag);
        cudaError_t launch_err = cudaGetLastError();
        cudaError_t sync_err = cudaDeviceSynchronize();
        float h_out;
        uint32_t h_diag[256];
        cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_diag, d_diag, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        float expected = 0.0f;
        for (int i = 0; i < 256; i++) expected += h_in[i];
        int nonzero = 0;
        for (int i = 0; i < 256; i++) if (h_diag[i] != 0u) nonzero++;
        printf("REP %d launch=%s sync=%s output=%f expected=%f match=%d nonzero_words=%d\n", rep,
               cudaGetErrorString(launch_err), cudaGetErrorString(sync_err), h_out, expected,
               (h_out == expected), nonzero);
    }
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_diag);
    return 0;
}
