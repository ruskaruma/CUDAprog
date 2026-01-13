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

__global__ void histogram(const int *data, int *hist, int n, int bins)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
    {
        int bin = data[idx] % bins;
        atomicAdd(&hist[bin], 1);
    }
}
int main()
{
    int N, bins;
    printf("Enter number of elements N: ");
    scanf("%d", &N);
    printf("Enter number of bins: ");
    scanf("%d", &bins);

    const size_t size = N * sizeof(int);
    const size_t hist_size = bins * sizeof(int);

    int *h_data = (int *)malloc(size);
    int *h_hist = (int *)malloc(hist_size);

    if (!h_data || !h_hist)
    {
        fprintf(stderr, "Host allocation failed\n");
        return 1;
    }

    printf("Enter %d integers for the data array:\n", N);
    for (int i = 0; i < N; i++)
    {
        scanf("%d", &h_data[i]);
    }

    int *d_data = nullptr;
    int *d_hist = nullptr;

    CHECK_CUDA(cudaMalloc(&d_data, size));
    CHECK_CUDA(cudaMalloc(&d_hist, hist_size));

    CHECK_CUDA(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(d_hist, 0, hist_size));

    const int threadsPerBlock = 256;
    const int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching %d blocks × %d threads\n",
           blocksPerGrid, threadsPerBlock);

    histogram<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_hist, N, bins);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(h_hist, d_hist, hist_size, cudaMemcpyDeviceToHost));

    printf("Histogram:\n");
    for (int i = 0; i < bins; i++)
    {
        printf("Bin %d: %d\n", i, h_hist[i]);
    }

    cudaFree(d_data);
    cudaFree(d_hist);

    free(h_data);
    free(h_hist);

    return 0;
}