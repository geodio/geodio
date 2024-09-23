#include "Addition.h"

// CPU-based addition
void addition_cpu(double* a, double* b, double* result, int size) {
    for (int i = 0; i < size; i++) {
        result[i] = a[i] + b[i];
    }
}

#ifdef USE_CUDA
// GPU-based addition (Stub)
void addition_gpu(double* a, double* b, double* result, int size) {
    // Implement CUDA here
}
#endif
