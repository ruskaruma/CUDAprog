#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

__global__ void divergenceKernel(float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        // Induce divergence within a warp
        if (idx % 2 == 0)
        {
            c[idx] = 100.0f;
        }
        else
        {
            c[idx] = 200.0f;
        }
    }
}

int main()
{
    const int N = 1024;
    const int size = N * sizeof(float);

    float *h_c = (float *)malloc(size);
    float *d_c;

    cudaMalloc(&d_c, size);

    // Initialize with 0
    cudaMemset(d_c, 0, size);

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    divergenceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, N);
    cudaDeviceSynchronize();

    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    bool success = true;
    for (int i = 0; i < N; i++)
    {
        float expected = (i % 2 == 0) ? 100.0f : 200.0f;
        if (h_c[i] != expected)
        {
            printf("Error at %d: got %f expected %f\n", i, h_c[i], expected);
            success = false;
            break;
        }
    }

    if (success)
    {
        printf("Divergence kernel pattern verified successfully.\n");
        printf("First 10 values:\n");
        for(int i=0; i<10; i++) printf("%f ", h_c[i]);
        printf("\n");
    }

    cudaFree(d_c);
    free(h_c);
    return 0;
}
