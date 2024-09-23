#include "LinearTransformation.h"
#include <stdlib.h>

// CPU-based Linear Transformation
void linear_transformation_cpu(double* in_data, double* weight, double* bias, double* result, int dim_out, int dim_in) {
    for (int i = 0; i < dim_out; i++) {
        result[i] = bias[i];
        for (int j = 0; j < dim_in; j++) {
            result[i] += weight[i * dim_in + j] * in_data[j];
        }
    }
}
