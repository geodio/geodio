#ifndef GEODIO_CPUBACKEND_H
#define GEODIO_CPUBACKEND_H

#include "Backend.h"
#include <cstring>

namespace dio {

    template<typename T>
    class CPUBackend : public Backend<T> {
    public:
        void allocate(T*& ptr, size_t size) override;
        void deallocate(T* ptr) override;
        void copy_to_device(T* device_ptr, const T* host_ptr, size_t size) override;
        void copy_to_host(T* host_ptr, const T* device_ptr, size_t size) override;
        void add(const T* a, const T* b, T* result, size_t size) override;
        void multiply(const T* a, const T* b, T* result, size_t size) override;
    };

} // namespace dio

#endif // GEODIO_CPUBACKEND_H
