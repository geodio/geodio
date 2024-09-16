#include "TensorGPU.h"
#ifdef USE_CUDA
#include <cuda_runtime.h>
#include <stdexcept>
#endif
#include <iostream>

namespace dio {
    #ifdef USE_CUDA

    // Constructor: initialize from initializer list (copy to GPU)
    template<typename T>
    TensorGPU<T>::TensorGPU(std::initializer_list<T> list) : Tensor<T>(list), gpu_data_(nullptr) {
        size_t total_size = Tensor<T>::compute_size(Tensor<T>::shape());
        cudaMalloc(&gpu_data_, total_size * sizeof(T));
        cudaMemcpy(gpu_data_, Tensor<T>::data_.data(), total_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Destructor: free GPU memory
    template<typename T>
    TensorGPU<T>::~TensorGPU() {
        if (gpu_data_) {
            cudaFree(gpu_data_);
        }
    }

    // Transfer data to GPU
    template<typename T>
    void TensorGPU<T>::to_gpu() {
        size_t total_size = Tensor<T>::compute_size(Tensor<T>::shape());
        cudaMemcpy(gpu_data_, Tensor<T>::data_.data(), total_size * sizeof(T), cudaMemcpyHostToDevice);
    }

    // Transfer data from GPU to CPU
    template<typename T>
    void TensorGPU<T>::from_gpu() {
        size_t total_size = Tensor<T>::compute_size(Tensor<T>::shape());
        cudaMemcpy(Tensor<T>::data_.data(), gpu_data_, total_size * sizeof(T), cudaMemcpyDeviceToHost);
    }

    #else

    // Fallback: no CUDA support, throw exception if used
    template<typename T>
    TensorGPU<T>::TensorGPU(std::initializer_list<T> list) : Tensor<T>(list) {
        throw std::runtime_error("CUDA support is not available on this system.");
    }

    template<typename T>
    TensorGPU<T>::~TensorGPU() = default;

    template<typename T>
    void TensorGPU<T>::to_gpu() {
        throw std::runtime_error("CUDA support is not available on this system.");
    }

    template<typename T>
    void TensorGPU<T>::from_gpu() {
        throw std::runtime_error("CUDA support is not available on this system.");
    }

    #endif

    // Explicit template instantiation
    template class TensorGPU<float>;
    template class TensorGPU<double>;
    template class TensorGPU<int>;
} // dio