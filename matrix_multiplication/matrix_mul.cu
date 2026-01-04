#include <stdio.h>
#include <cuda_runtime.h>

#define WIDTH 16

// CUDA kernel for matrix multiplication
__global__ void matrixMul(const float *a, const float *b, float *c, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += a[row * width + k] * b[k * width + col];
        }
        c[row * width + col] = sum;
    }
}

int main() {
    const int size = WIDTH * WIDTH * sizeof(float);
    
    // Allocate host memory
    float *h_a = (float*)malloc(size);
    float *h_b = (float*)malloc(size);
    float *h_c = (float*)malloc(size);
    
    if (!h_a || !h_b || !h_c) {
        printf("Failed to allocate host memory\n");
        return 1;
    }
    
    // Initialize matrices
    printf("Initializing %dx%d matrices...\n", WIDTH, WIDTH);
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        h_a[i] = 1.0f;
        h_b[i] = 2.0f;
    }
    
    // Allocate device memory
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    if (!d_a || !d_b || !d_c) {
        printf("Failed to allocate device memory\n");
        free(h_a); free(h_b); free(h_c);
        return 1;
    }
    
    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel with 2D grid
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((WIDTH + 15) / 16, (WIDTH + 15) / 16);
    matrixMul<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, WIDTH);
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Verify result (each element should be WIDTH * 1.0 * 2.0 = WIDTH * 2.0)
    float expected = WIDTH * 2.0f;
    bool success = true;
    for (int i = 0; i < WIDTH * WIDTH; i++) {
        if (h_c[i] != expected) {
            printf("Error at index %d: %f != %f\n", i, h_c[i], expected);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Matrix multiplication successful!\n");
        printf("Sample results (first 3x3 submatrix):\n");
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                printf("%6.1f ", h_c[i * WIDTH + j]);
            }
            printf("\n");
        }
    }
    
    // Free memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    free(h_a);
    free(h_b);
    free(h_c);
    
    return 0;
}
