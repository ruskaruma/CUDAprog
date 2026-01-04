# Matrix Multiplication CUDA Program

This program demonstrates parallel matrix multiplication using CUDA.

## Description
The program multiplies two 16x16 matrices using GPU parallelization with 2D thread blocks.

## Compilation
```bash
nvcc matrix_mul.cu -o matrix_mul
```

## Execution
```bash
./matrix_mul
```

## Output
The program will verify the results and display a 3x3 submatrix of the result.
