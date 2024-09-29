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
#ifndef GEODIO_CPUBACKEND_H
#define GEODIO_CPUBACKEND_H

#include "Backend.h"
#include <functional>

namespace dio {

    template<typename T>
    class CPUBackend : public Backend<T> {
    public:
        void allocate(T*& ptr, size_t size) override;
        void deallocate(T* ptr) override;
        void copy_to_device(T* device_ptr, const T* host_ptr, size_t size) override;
        void copy_to_host(T* host_ptr, const T* device_ptr, size_t size) override;

        template<typename U, typename R>
        void matmul(const T* a, const U* b, R* result,
                               size_t m, size_t n, size_t k,
                               const std::vector<size_t>& strides_a,
                               const std::vector<size_t>& strides_b);

        // Virtual function for elementwise operation with generic pointers
        void elementwise_operation_generic(const void* a, const void* b, void* result,
                                                   std::function<void(const void*, const void*, void*)> func,
                                                   size_t total_size, const std::vector<size_t>& result_shape,
                                                   const std::vector<size_t>& adjusted_strides1,
                                                   const std::vector<size_t>& adjusted_strides2) override;
        // Generic element-wise operation between two tensors
        #pragma warning(push)
        #pragma warning(disable:26434)
        template<typename U, typename R>
        void elementwise_operation(const T* a, const U* b, R* result, std::function<R(T, U)> func,
                                                  const size_t total_size, const std::vector<size_t>& result_shape,
                                                  const std::vector<size_t>& adjusted_strides1,
                                                  const std::vector<size_t>& adjusted_strides2);
        #pragma warning(pop)

        template<typename R>
        void apply_unary_function(const T* a, R* result, std::function<R(T)> func, size_t size);
    };

} // namespace dio

#endif // GEODIO_CPUBACKEND_H
