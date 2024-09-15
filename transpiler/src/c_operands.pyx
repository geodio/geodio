# Import necessary Cython modules
from cpython.object cimport PyObject
from libc.stdlib cimport malloc, free
import numpy as np

# Declare external C functions
cdef extern from "LinearTransformation.h":
    void linear_transformation_cpu(double* in_data, double* weight, double* bias, double* result, int dim_out, int dim_in)
    # TODO add support for multiple targets and multiple data types that are
    #  inferred dynamically
    # void linear_transformation_gpu(double * in_data, double * weight,
    #                                double * bias, double * result, int dim_out,
    #                                int dim_in)  # GPU version

cdef extern from "Addition.h":
    void addition_cpu(double* a, double* b, double* result, int size)

# Helper function to flatten and convert inputs to C arrays
cdef double* convert_to_c_array(object input, int size):
    cdef double* c_array = <double*> malloc(size * sizeof(double))

    if isinstance(input, np.ndarray):
        input = np.ascontiguousarray(input)
        for i in range(size):
            c_array[i] = input.flat[i]  # Flattening arbitrary-dimension NumPy array
    elif isinstance(input, list):
        flat_list = flatten_list(input)  # Flattening list of any dimension
        for i in range(size):
            c_array[i] = flat_list[i]
    elif isinstance(input, (int, float)):
        for i in range(size):
            c_array[i] = input
    else:
        raise TypeError("Unsupported input type. Expected NumPy array, list, int, or float.")

    return c_array

# Function to flatten any-dimensional lists
cdef list flatten_list(list input_list):
    flat_list = []
    for item in input_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))  # Recursively flatten lists
        else:
            flat_list.append(item)
    return flat_list


# LinearTransformationWrapper class
cdef class LinearTransformationWrapper:
    cdef int dim_in, dim_out
    cdef double* in_data
    cdef double* weight
    cdef double* bias
    cdef double* result
    cdef bint use_gpu

    def __init__(self, int dim_in, int dim_out, bint use_gpu=False):
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.in_data = NULL
        self.weight = NULL
        self.bias = NULL
        self.result = <double*> malloc(dim_out * sizeof(double))
        self.use_gpu = use_gpu

    # Dynamically set input data
    def set_input(self, object in_data):
        self.in_data = convert_to_c_array(in_data, self.dim_in)

    # Dynamically set weights
    def set_weights(self, object weight):
        self.weight = convert_to_c_array(weight, self.dim_in * self.dim_out)

    # Dynamically set bias
    def set_bias(self, object bias):
        self.bias = convert_to_c_array(bias, self.dim_out)

    def linear_transform(self):
        if self.in_data == NULL or self.weight == NULL or self.bias == NULL:
            raise ValueError("Input data, weight, and bias must be set before calling linear_transform.")
        if self.use_gpu:
            pass
            # linear_transformation_gpu(self.in_data, self.weight, self.bias,
            #                           self.result, self.dim_out, self.dim_in)
        else:
            linear_transformation_cpu(self.in_data, self.weight, self.bias,
                                      self.result, self.dim_out, self.dim_in)

    def get_result(self):
        return [self.result[i] for i in range(self.dim_out)]

    def __dealloc__(self):
        if self.in_data != NULL:
            free(self.in_data)
        if self.weight != NULL:
            free(self.weight)
        if self.bias != NULL:
            free(self.bias)
        if self.result != NULL:
            free(self.result)


# AdditionWrapper class
cdef class AdditionWrapper:
    cdef double* a
    cdef double* b
    cdef double* result
    cdef int size

    def __init__(self, int size):
        self.size = size
        self.a = NULL
        self.b = NULL
        self.result = <double*> malloc(size * sizeof(double))

    # Dynamically set operands
    def set_operands(self, object a, object b):
        self.a = convert_to_c_array(a, self.size)
        self.b = convert_to_c_array(b, self.size)

    def add(self):
        if self.a == NULL or self.b == NULL:
            raise ValueError("Both operands must be set before calling add.")
        addition_cpu(self.a, self.b, self.result, self.size)

    def get_result(self):
        return [self.result[i] for i in range(self.size)]

    def __dealloc__(self):
        if self.a != NULL:
            free(self.a)
        if self.b != NULL:
            free(self.b)
        if self.result != NULL:
            free(self.result)
