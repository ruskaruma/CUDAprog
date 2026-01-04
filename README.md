# CUDA Programming Examples

This repository contains simple CUDA programs demonstrating GPU parallel computing.

## Programs

### 1. Vector Addition
Location: `vector_addition/`

A simple program that demonstrates parallel vector addition using CUDA. Adds two vectors of 1024 floating-point numbers on the GPU.

**Features:**
- Basic CUDA kernel implementation
- Memory management (host and device)
- Result verification

### 2. Matrix Multiplication
Location: `matrix_multiplication/`

A program demonstrating 2D parallel matrix multiplication using CUDA. Multiplies two 16x16 matrices on the GPU.

**Features:**
- 2D thread block configuration
- Matrix operations on GPU
- Result verification

## Prerequisites

- NVIDIA GPU with CUDA support
- CUDA Toolkit (nvcc compiler)

## Quick Start

Each program directory contains its own README with compilation and execution instructions.

```bash
# Vector Addition
cd vector_addition
nvcc vector_add.cu -o vector_add
./vector_add

# Matrix Multiplication
cd matrix_multiplication
nvcc matrix_mul.cu -o matrix_mul
./matrix_mul
```

## License

See LICENSE file for details.
