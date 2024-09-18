#include "CPUBackend.h"
#include "Tensor.h"
#include <cstring>
#include <functional>
#include <stdexcept>

namespace dio {

    // Allocates memory for a tensor
    template<typename T>
    void CPUBackend<T>::allocate(T*& ptr, size_t size) {
        ptr = new T[size];
    }

    // Deallocates memory of a tensor
    template<typename T>
    void CPUBackend<T>::deallocate(T* ptr) {
        delete[] ptr;
    }

    // Copy tensor data from host to device (in this case, both are CPU so it's just a memory copy)
    template<typename T>
    void CPUBackend<T>::copy_to_device(T* device_ptr, const T* host_ptr, size_t size) {
        std::memcpy(device_ptr, host_ptr, size * sizeof(T));
    }

    // Copy tensor data from device to host (same as above)
    template<typename T>
    void CPUBackend<T>::copy_to_host(T* host_ptr, const T* device_ptr, size_t size) {
        std::memcpy(host_ptr, device_ptr, size * sizeof(T));
    }

    // Matrix multiplication of two tensors
    template<typename T>
    void CPUBackend<T>::matmul(const T* a, const T* b, T* result, size_t m,size_t n,size_t k) {
        #pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                result[i * k + j] = 0;
                for (size_t p = 0; p < n; ++p) {
                    result[i * k + j] += a[i * n + p] * b[p * k + j];
                }
            }
        }
    }

    template<typename T>
    template<typename U, typename R>
    void CPUBackend<T>::elementwise_operation(const T* a, const U* b, R* result,
                                          std::function<R(T, U)> func,
                                          size_t total_size, const std::vector<size_t>& result_shape,
                                          const std::vector<size_t>& adjusted_strides1,
                                          const std::vector<size_t>& adjusted_strides2) {
    // Perform the operation
    #pragma omp parallel for
    for (size_t i = 0; i < total_size; ++i) {
        size_t idx1 = 0, idx2 = 0;
        size_t idx = i;
        for (int dim = static_cast<int>(result_shape.size()) - 1; dim >= 0; --dim) {
            size_t index = idx % result_shape[dim];
            idx /= result_shape[dim];
            idx1 += adjusted_strides1[dim] * index;
            idx2 += adjusted_strides2[dim] * index;
        }
        result[i] = func(a[idx1], b[idx2]);
    }
}

template<typename T>
void CPUBackend<T>::apply_unary_function(const T* a, T* result, std::function<T(T)> func, size_t size) {
    #pragma omp parallel for
    for (size_t i = 0; i < size; ++i) {
        result[i] = func(a[i]);
    }
}


    // Implementation of the virtual function using generic pointers
    template<typename T>
    void CPUBackend<T>::elementwise_operation_generic(const void* a, const void* b, void* result,
                                                      std::function<void(const void*, const void*, void*)> func,
                                                      size_t total_size, const std::vector<size_t>& result_shape,
                                                      const std::vector<size_t>& adjusted_strides1,
                                                      const std::vector<size_t>& adjusted_strides2) {
        // Perform the operation
        #pragma omp parallel for
        for (size_t i = 0; i < total_size; ++i) {
            size_t idx1 = 0, idx2 = 0;
            size_t idx = i;
            for (size_t dim = 0; dim < result_shape.size(); ++dim) {
                size_t index = idx % result_shape[result_shape.size() - 1 - dim];
                idx /= result_shape[result_shape.size() - 1 - dim];
                idx1 += adjusted_strides1[adjusted_strides1.size() - 1 - dim] * index;
                idx2 += adjusted_strides2[adjusted_strides2.size() - 1 - dim] * index;
            }
            func(static_cast<const void*>(a), static_cast<const void*>(b), static_cast<void*>(result));
        }
    }

    // Explicit template instantiation
    template class CPUBackend<float>;
    template class CPUBackend<double>;
    template class CPUBackend<int>;

} // namespace dio
