CUDA Programming Examples

This repository contains CUDA programming examples for learning GPU computing.

Directories:
- basic_and_execution_model: Basic vector operations and execution patterns.
- kernels: Simple kernel examples for vector sum and subtraction.
- memory_heirarchy: Matrix multiplication and vector dot product.
- parallelPatterns: Histogram computation.
- cuda_device_info: Tool to query CUDA device information.
- cuda_parallel_sum: Tool to compute sum of numbers using parallel reduction.

Compile with: nvcc -o output file.cu
Run: ./output