#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                     \
    do                                                       \
    {                                                        \
        cudaError_t err = call;                              \
        if (err != cudaSuccess)                              \
        {                                                    \
            fprintf(stderr, "CUDA error %s:%d: %s\n",        \
                    __FILE__, __LINE__,                      \
                    cudaGetErrorString(err));                \
            exit(1);                                         \
        }                                                    \
    } while (0)

__global__ void vectorScale(const float *a, float scalar, float *c, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] * scalar;
}

int main()
{
    int N;
    float scalar;
    printf("Enter vector size N: ");
    scanf("%d", &N);
    printf("Enter scalar value: ");
    scanf("%f", &scalar);

    const size_t size = N * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_c = (float *)malloc(size);

    if (!h_a || !h_c)
    {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    printf("Enter %d elements for vector A:\n", N);
    for (int i = 0; i < N; i++)
    {
        scanf("%f", &h_a[i]);
    }

    float *d_a = nullptr;
    float *d_c = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_c, size));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching %d blocks × %d threads\n",
           blocksPerGrid, threadsPerBlock);

    vectorScale<<<blocksPerGrid, threadsPerBlock>>>(d_a, scalar, d_c, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost));

    printf("Result vector C (A * %.2f):\n", scalar);
    for (int i = 0; i < N; i++)
    {
        printf("%.2f ", h_c[i]);
    }
    printf("\n");

    cudaFree(d_a);
    cudaFree(d_c);

    free(h_a);
    free(h_c);

    return 0;
}