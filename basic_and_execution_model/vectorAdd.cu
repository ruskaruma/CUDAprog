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
    const int N = 1024;
    const int size = N * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    if (!h_a || !h_b || !h_c)
    {
        printf("Failed to allocate host memory\n");
        return 1;
    }

    for (int i = 0; i < N; i++)
    {
        h_a[i] = (float)i;
        h_b[i] = (float)(2 * i);
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

    bool success = true;
    for (int i = 0; i < N; i++)
    {
        if (h_c[i] != h_a[i] + h_b[i])
        {
            printf("Error at index %d: %f != %f\n", i, h_c[i], h_a[i] + h_b[i]);
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("Vector addition successful! First 5 results:\n");
        for (int i = 0; i < 5; i++)
        {
            printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
        }
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    free(h_a);
    free(h_b);
    free(h_c);

    return 0;
}
