#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                     \
    do                                                       \
    {                                                        \
        cudaError_t err = call;                              \
        if(err != cudaSuccess)                              \
        {                                                    \
            printf("CUDA error %s:%d: %s\n",                 \
                   __FILE__, __LINE__,                       \
                   cudaGetErrorString(err));                 \
            exit(1);                                         \
        }                                                    \
    } while (0)

__global__ void divergenceKernel(float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
    {
        return;
    }
    if ((idx & 1) == 0)
        c[idx] = 100.0f;
    else
        c[idx] = 200.0f;
}
int main()
{
    const int N = 1024;
    const size_t size = N * sizeof(float);

    float *h_c = (float *)malloc(size);
    if(!h_c)
    {
        printf("Host allocation failed\n");
        return 1;
    }

    float *d_c = nullptr;
    CHECK_CUDA(cudaMalloc(&d_c, size));
    CHECK_CUDA(cudaMemset(d_c, 0, size));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    divergenceKernel<<<blocksPerGrid, threadsPerBlock>>>(d_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++)
    {
        float expected = (i & 1) ? 200.0f : 100.0f;
        if (h_c[i] != expected)
        {
            printf("Mismatch at %d: got %f expected %f\n",
                   i, h_c[i], expected);
            return 1;
        }
    }

    printf("Warp divergence pattern verified.\n");
    printf("First 10 values:\n");
    for (int i = 0; i < 10; i++)
        printf("%0.1f ", h_c[i]);
    printf("\n");

    cudaFree(d_c);
    free(h_c);
    return 0;
}
