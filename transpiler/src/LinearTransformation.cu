#include <cuda_runtime.h>
#include "LinearTransformation.h"

// CUDA kernel for matrix multiplication (used in linear transformation)
__global__ void matmul(double* in_data, double* weight, double* bias, double* result, int dim_out, int dim_in) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < dim_out) {
        double sum = bias[row];
        for (int i = 0; i < dim_in; i++) {
            sum += weight[row * dim_in + i] * in_data[i];
        }
        result[row] = sum;
    }
}

// Host function that launches the CUDA kernel
void linear_transformation_gpu(double* in_data, double* weight, double* bias, double* result, int dim_out, int dim_in) {
    double* d_in_data;
    double* d_weight;
    double* d_bias;
    double* d_result;

    // Allocate memory on the device
    cudaMalloc((void**)&d_in_data, dim_in * sizeof(double));
    cudaMalloc((void**)&d_weight, dim_in * dim_out * sizeof(double));
    cudaMalloc((void**)&d_bias, dim_out * sizeof(double));
    cudaMalloc((void**)&d_result, dim_out * sizeof(double));

    // Copy data to the device
    cudaMemcpy(d_in_data, in_data, dim_in * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weight, weight, dim_in * dim_out * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, bias, dim_out * sizeof(double), cudaMemcpyHostToDevice);

    // Launch the kernel (1 block with dim_out threads)
    dim3 threadsPerBlock(1, 256);
    dim3 blocksPerGrid(1, (dim_out + 255) / 256);
    matmul<<<blocksPerGrid, threadsPerBlock>>>(d_in_data, d_weight, d_bias, d_result, dim_out, dim_in);

    // Copy result back to host
    cudaMemcpy(result, d_result, dim_out * sizeof(double), cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in_data);
    cudaFree(d_weight);
    cudaFree(d_bias);
    cudaFree(d_result);
}
