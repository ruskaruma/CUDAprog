#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void sumKernel(float *d_data, float *d_sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        atomicAdd(d_sum, d_data[idx]);
    }
}

int main(int argc, char *argv[]) {
    if (argc < 2) {
        printf("Usage: %s num1 num2 ...\n", argv[0]);
        return 1;
    }
    int n = argc - 1;
    float *h_data = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; i++) {
        h_data[i] = atof(argv[i + 1]);
    }
    float *d_data;
    float *d_sum;
    cudaMalloc(&d_data, n * sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));
    cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice);
    int threadsPerBlock = 256;
    int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    sumKernel<<<blocks, threadsPerBlock>>>(d_data, d_sum, n);
    float h_sum;
    cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
    printf("Sum: %f\n", h_sum);
    cudaFree(d_data);
    cudaFree(d_sum);
    free(h_data);
    return 0;
}