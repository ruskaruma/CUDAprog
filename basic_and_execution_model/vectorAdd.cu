#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void vectorAdd(const float *a, const float *b, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        c[idx] = a[idx] + b[idx];
    }
}

int main()
{
    int N;
    printf("Enter vector size N: ");
    scanf("%d", &N);
    const int size = N * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    if (!h_a || !h_b || !h_c)
    {
        printf("Failed to allocate host memory\n");
        return 1;
    }

    printf("Enter %d elements for vector A:\n", N);
    for (int i = 0; i < N; i++)
    {
        scanf("%f", &h_a[i]);
    }

    printf("Enter %d elements for vector B:\n", N);
    for (int i = 0; i < N; i++)
    {
        scanf("%f", &h_b[i]);
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;

    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    if (!d_a || !d_b || !d_c)
    {
        printf("Failed to allocate device memory\n");
        return 1;
    }

    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    printf("Result vector C (A + B):\n");
    for (int i = 0; i < N; i++)
    {
        printf("%.2f ", h_c[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
