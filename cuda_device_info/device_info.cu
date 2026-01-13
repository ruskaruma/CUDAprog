#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Number of CUDA devices: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);
        printf("Device %d: %s\n", i, prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        printf("Total global memory: %lu bytes\n", prop.totalGlobalMem);
        printf("Shared memory per block: %lu bytes\n", prop.sharedMemPerBlock);
        printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
        printf("Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Number of multiprocessors: %d\n", prop.multiProcessorCount);
    }
    return 0;
}