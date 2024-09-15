#ifndef ADDITION_H
#define ADDITION_H

void addition_cpu(double* a, double* b, double* result, int size);
#ifdef USE_CUDA
void addition_gpu(double* a, double* b, double* result, int size);
#endif

#endif
