//    Geodio Neural Network Framework - A lightweight neural network library focused on custom architecture design and optimization.
//    Copyright (C) Copyright (C) 2024 Geodio (created by Rareș Polenciuc)
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#include "CPUBackend.h"
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
    template<typename U, typename R>
    void CPUBackend<T>::matmul(const T* a, const U* b, R* result,
                               size_t m, size_t n, size_t k,
                               const std::vector<size_t>& strides_a,
                               const std::vector<size_t>& strides_b) {
        // A is (m x n), B is (n x k), result is (m x k)

        #pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < k; ++j) {
                R sum = 0;
                for (size_t p = 0; p < n; ++p) {
                    // Compute indices for A and B, considering strides and possible broadcasting
                    size_t idx_a = i * strides_a[0] + p * strides_a[1];
                    size_t idx_b = p * strides_b[0] + j * strides_b[1];

                    sum += a[idx_a] * b[idx_b];
                }
                result[i * k + j] = sum;
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
template<typename R>
void CPUBackend<T>::apply_unary_function(const T* a, R* result, std::function<R(T)> func, size_t size) {
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
