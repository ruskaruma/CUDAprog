#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                     \
    do                                                       \
    {                                                        \
        cudaError_t err = call;                              \
        if (err != cudaSuccess)                              \
        {                                                    \
            fprintf(stderr, "CUDA error %s:%d: %s\n",__FILE__, __LINE__,cudaGetErrorString(err)); \
            exit(1);                                         \
        }                                                    \
    } while (0)

__global__ void atomicAddKernel(int *d_count, int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        atomicAdd(d_count, 1);
}

int main()
{
    int N;
    printf("Enter number of threads to launch: ");
    scanf("%d", &N);
    int h_count = 0;
    int *d_count = nullptr;
    CHECK_CUDA(cudaMalloc(&d_count, sizeof(int)));
    CHECK_CUDA(cudaMemcpy(d_count, &h_count, sizeof(int), cudaMemcpyHostToDevice));
    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    const int totalThreads = blocksPerGrid * threadsPerBlock;
    atomicAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_count, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&h_count, d_count, sizeof(int), cudaMemcpyDeviceToHost));
    printf("Launched %d blocks × %d threads\n", blocksPerGrid, threadsPerBlock);
    printf("Total threads launched: %d\n", totalThreads);
    printf("Atomic add result: %d\n", h_count);
    if(h_count == N)
    {
        printf("Success: atomic updates serialized correctly.\n");
    }
    else
    {
        printf("Mismatch: expected %d\n", N);
    }
    cudaFree(d_count);
    return 0;
}
