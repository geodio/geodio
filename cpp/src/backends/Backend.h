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
#ifndef GEODIO_BACKEND_H
#define GEODIO_BACKEND_H

#include <cstddef>
#include <functional>

namespace dio {

    template<typename T>
    class Backend {
    public:
        virtual ~Backend() = default;

        // Memory management
        virtual void allocate(T*& ptr, size_t size) = 0;
        virtual void deallocate(T* ptr) = 0;

        // Data transfer
        virtual void copy_to_device(T* device_ptr, const T* host_ptr, size_t size) = 0;
        virtual void copy_to_host(T* host_ptr, const T* device_ptr, size_t size) = 0;

        // Numerical operations between two tensors
        template<typename U, typename R>
        void matmul(const T* a, const U* b, R* result,
                               size_t m, size_t n, size_t k,
                               const std::vector<size_t>& strides_a,
                               const std::vector<size_t>& strides_b);

        template<typename U>
        void apply_unary_function(const T* a, U* result, std::function<U(T)> func, size_t size);

        // Virtual function for elementwise operation with generic pointers
        virtual void elementwise_operation_generic(const void* a, const void* b, void* result,
                                                   std::function<void(const void*, const void*, void*)> func,
                                                   size_t total_size, const std::vector<size_t>& result_shape,
                                                   const std::vector<size_t>& adjusted_strides1,
                                                   const std::vector<size_t>& adjusted_strides2) = 0;

        // Template function to be called within elementwise_operation_generic
        template<typename U, typename R>
        void elementwise_operation(const T* a, const U* b, R* result, std::function<R(T, U)> func,
                                   size_t total_size, const std::vector<size_t>& result_shape,
                                   const std::vector<size_t>& adjusted_strides1,
                                   const std::vector<size_t>& adjusted_strides2);
    };

    template<typename T>
    template<typename U, typename R>
    void Backend<T>::elementwise_operation(const T* a, const U* b, R* result, std::function<R(T, U)> func,
                                              const size_t total_size, const std::vector<size_t>& result_shape,
                                              const std::vector<size_t>& adjusted_strides1,
                                              const std::vector<size_t>& adjusted_strides2) {
        // Perform the operation
        for (size_t i = 0; i < total_size; ++i) {
            size_t idx1 = 0, idx2 = 0;
            size_t idx = i;
            for (size_t dim = 0; dim < result_shape.size(); ++dim) {
                size_t index = idx % result_shape[result_shape.size() - 1 - dim];
                idx /= result_shape[result_shape.size() - 1 - dim];
                idx1 += adjusted_strides1[adjusted_strides1.size() - 1 - dim] * index;
                idx2 += adjusted_strides2[adjusted_strides2.size() - 1 - dim] * index;
            }
            result[i] = func(a[idx1], b[idx2]);
        }
    }

    template<typename T>
    template<typename U, typename R>
    void Backend<T>::matmul(const T* a, const U* b, R* result,
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
    template<typename U>
    void Backend<T>::apply_unary_function(const T* a, U* result, std::function<U(T)> func, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            result[i] = func(a[i]);
        }
    }

} // namespace dio

#endif // GEODIO_BACKEND_H
