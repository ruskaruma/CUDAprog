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

__global__ void vectorSum(const float *a, float *sum, int n)
{
    __shared__ float sdata[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    sdata[tid] = (idx < n) ? a[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(sum, sdata[0]);
    }
}

int main()
{
    int N;
    printf("Enter vector size N: ");
    scanf("%d", &N);

    const size_t size = N * sizeof(float);

    float *h_a = (float *)malloc(size);

    if (!h_a)
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
    float *d_sum = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(float)));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching %d blocks × %d threads\n",
           blocksPerGrid, threadsPerBlock);

    vectorSum<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_sum, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_sum;
    CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Sum of vector elements: %.2f\n", h_sum);

    cudaFree(d_a);
    cudaFree(d_sum);

    free(h_a);

    return 0;
}