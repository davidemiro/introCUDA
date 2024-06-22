#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel for matrix addition
__global__ void matrixAdd(float *A, float *B, float *C, int numRows, int numCols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < numRows && col < numCols) {
        int idx = row * numCols + col;
        C[idx] = A[idx] + B[idx];
    }
}

void initializeMatrix(float *matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows * numCols; ++i) {
        matrix[i] = static_cast<float>(i);
    }
}

void printMatrix(const float *matrix, int numRows, int numCols) {
    for (int i = 0; i < numRows; ++i) {
        for (int j = 0; j < numCols; ++j) {
            std::cout << matrix[i * numCols + j] << " ";
        }
        std::cout << std::endl;
    }
}

int main() {
    // Define matrix dimensions
    const int numRows = 4;
    const int numCols = 4;
    const int matrixSize = numRows * numCols * sizeof(float);

    // Allocate host memory
    float *h_A = (float *)malloc(matrixSize);
    float *h_B = (float *)malloc(matrixSize);
    float *h_C = (float *)malloc(matrixSize);

    // Initialize matrices
    initializeMatrix(h_A, numRows, numCols);
    initializeMatrix(h_B, numRows, numCols);

    // Allocate device memory
    float *d_A, *d_B, *d_C;
    cudaMalloc((void **)&d_A, matrixSize);
    cudaMalloc((void **)&d_B, matrixSize);
    cudaMalloc((void **)&d_C, matrixSize);

    // Copy matrices from host to device
    cudaMemcpy(d_A, h_A, matrixSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, matrixSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((numCols + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (numRows + threadsPerBlock.y - 1) / threadsPerBlock.y);

    // Launch the kernel
    matrixAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numRows, numCols);

    // Copy result matrix from device to host
    cudaMemcpy(h_C, d_C, matrixSize, cudaMemcpyDeviceToHost);

    // Print the result
    std::cout << "Matrix A:" << std::endl;
    printMatrix(h_A, numRows, numCols);
    std::cout << "Matrix B:" << std::endl;
    printMatrix(h_B, numRows, numCols);
    std::cout << "Matrix C (A + B):" << std::endl;
    printMatrix(h_C, numRows, numCols);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
