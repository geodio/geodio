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
        virtual void matmul(const T* a, const T* b, T* result, size_t m,size_t n,size_t k) = 0;

        virtual void apply_unary_function(const T* a, T* result, std::function<T(T)> func, size_t size) = 0;

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

} // namespace dio

#endif // GEODIO_BACKEND_H
