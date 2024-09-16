#ifndef GEODIO_BACKEND_H
#define GEODIO_BACKEND_H

#include <cstddef>

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

        // Numerical operations
        virtual void add(const T* a, const T* b, T* result, size_t size) = 0;
        virtual void multiply(const T* a, const T* b, T* result, size_t size) = 0;

        // More methods as needed
    };

} // namespace dio

#endif // GEODIO_BACKEND_H
