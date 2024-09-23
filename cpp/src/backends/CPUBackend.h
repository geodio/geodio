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

        template<typename U>
        void matmul(const T* a, const U* b, typename std::common_type<T, U>::type* result, size_t m,size_t n,size_t k);

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
