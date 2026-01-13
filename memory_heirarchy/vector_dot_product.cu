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

__global__ void dotProduct(const float *a, const float *b, float *result, int n)
{
    __shared__ float temp[256];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    temp[tid] = (idx < n) ? a[idx] * b[idx] : 0.0f;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        if (tid < s)
        {
            temp[tid] += temp[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0)
    {
        atomicAdd(result, temp[0]);
    }
}

int main()
{
    int N;
    printf("Enter vector size N: ");
    scanf("%d", &N);

    const size_t size = N * sizeof(float);

    float *h_a = (float *)malloc(size);
    float *h_b = (float *)malloc(size);

    if (!h_a || !h_b)
    {
        fprintf(stderr, "Host allocation failed\n");
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
    float *d_result = nullptr;

    CHECK_CUDA(cudaMalloc(&d_a, size));
    CHECK_CUDA(cudaMalloc(&d_b, size));
    CHECK_CUDA(cudaMalloc(&d_result, sizeof(float)));

    CHECK_CUDA(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_result, 0, sizeof(float)));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching %d blocks × %d threads\n",
           blocksPerGrid, threadsPerBlock);

    dotProduct<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_result, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    float h_result;
    CHECK_CUDA(cudaMemcpy(&h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost));

    printf("Dot product of vectors: %.2f\n", h_result);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_result);

    free(h_a);
    free(h_b);

    return 0;
}