#include "CPUBackend.h"

namespace dio {

    template<typename T>
    void CPUBackend<T>::allocate(T*& ptr, size_t size) {
        ptr = new T[size];
    }

    template<typename T>
    void CPUBackend<T>::deallocate(T* ptr) {
        delete[] ptr;
    }

    template<typename T>
    void CPUBackend<T>::copy_to_device(T* device_ptr, const T* host_ptr, size_t size) {
        std::memcpy(device_ptr, host_ptr, size * sizeof(T));
    }

    template<typename T>
    void CPUBackend<T>::copy_to_host(T* host_ptr, const T* device_ptr, size_t size) {
        std::memcpy(host_ptr, device_ptr, size * sizeof(T));
    }

    template<typename T>
    void CPUBackend<T>::add(const T* a, const T* b, T* result, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] + b[i];
        }
    }

    template<typename T>
    void CPUBackend<T>::multiply(const T* a, const T* b, T* result, size_t size) {
        #pragma omp parallel for
        for (size_t i = 0; i < size; ++i) {
            result[i] = a[i] * b[i];
        }
    }


    template<typename T>
    void CPUBackend<T>::matmul(const T* a, const T* b, T* result, size_t m,size_t n,size_t k) {
        #pragma omp parallel for
        for (size_t i = 0; i < m; ++i) {
            result[i] = a[i] * b[i];
        }
    }

    // Explicit template instantiation
    template class CPUBackend<float>;
    template class CPUBackend<double>;
    template class CPUBackend<int>;

} // namespace dio
