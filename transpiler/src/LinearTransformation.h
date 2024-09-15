#ifndef LINEAR_TRANSFORMATION_H
#define LINEAR_TRANSFORMATION_H

// CPU-based Linear Transformation
void linear_transformation_cpu(double* in_data, double* weight, double* bias, double* result, int dim_out, int dim_in);

// GPU-based Linear Transformation
#ifdef USE_CUDA
void linear_transformation_gpu(double* in_data, double* weight, double* bias, double* result, int dim_out, int dim_in);
#endif

#endif
