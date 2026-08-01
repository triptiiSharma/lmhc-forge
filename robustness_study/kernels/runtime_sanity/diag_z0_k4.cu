#include <stdint.h>
#include <stdio.h>
#include <cuda_runtime.h>

__launch_bounds__(256, 1)
__global__ void diag_z0_k4(const float *A, const float *B, float *C, int M, int N, int Kdim, uint32_t *diag_out)
{
    int tid = threadIdx.x;
    int tx = tid % 16;
    int ty = tid / 16;
    int row = blockIdx.y * 16 + ty;
    int col = blockIdx.x * 16 + tx;
    __shared__ float sbuf[256];
    __shared__ float b_tile[256];
    float acc = 0.0f;
    for (int t = 0; t < (Kdim + 15) / 16; t++) {
        int a_col = t * 16 + tx;
        int b_row = t * 16 + ty;
        sbuf[ty * 16 + tx] = (row < M && a_col < Kdim) ? A[row * Kdim + a_col] : 0.0f;
        b_tile[ty * 16 + tx] = (b_row < Kdim && col < N) ? B[b_row * N + col] : 0.0f;
        __syncthreads();
        for (int k = 0; k < 16; k++)
            acc += sbuf[ty * 16 + k] * b_tile[k * 16 + tx];
        __syncthreads();
    }
    for (int i = tid; i < 256; i += 256)
        diag_out[i] = __float_as_uint(sbuf[i]);
    if (row < M && col < N)
        C[row * N + col] = acc;
}

int main() {
    float *dA, *dB, *dC; uint32_t *d_diag;
    float hA[256], hB[256];
    for (int i = 0; i < 256; i++) { hA[i] = (float)(i % 7); hB[i] = (float)(i % 5); }
    cudaMalloc(&dA, 256 * sizeof(float));
    cudaMalloc(&dB, 256 * sizeof(float));
    cudaMalloc(&dC, 256 * sizeof(float));
    cudaMalloc(&d_diag, 256 * sizeof(uint32_t));
    cudaMemcpy(dA, hA, 256 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB, 256 * sizeof(float), cudaMemcpyHostToDevice);
    for (int rep = 0; rep < 5; rep++) {
        diag_z0_k4<<<1, 256>>>(dA, dB, dC, 16, 16, 16, d_diag);
        cudaError_t launch_err = cudaGetLastError();
        cudaError_t sync_err = cudaDeviceSynchronize();
        float hC[256];
        uint32_t h_diag[256];
        cudaMemcpy(hC, dC, 256 * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_diag, d_diag, 256 * sizeof(uint32_t), cudaMemcpyDeviceToHost);
        float expected[256];
        int ok = 1;
        for (int r = 0; r < 16; r++) for (int c = 0; c < 16; c++) {
            float acc = 0.0f;
            for (int k = 0; k < 16; k++) acc += hA[r*16+k] * hB[k*16+c];
            expected[r*16+c] = acc;
            if (hC[r*16+c] != acc) ok = 0;
        }
        int nonzero = 0;
        for (int i = 0; i < 256; i++) if (h_diag[i] != 0u) nonzero++;
        printf("REP %d launch=%s sync=%s gemm_correct=%d nonzero_words=%d\n", rep,
               cudaGetErrorString(launch_err), cudaGetErrorString(sync_err), ok, nonzero);
    }
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(d_diag);
    return 0;
}
