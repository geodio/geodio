# file_operations.pyx
# Import the necessary Cython definitions
from cpython.object cimport PyObject

# Declare external C functions
cdef extern from "FileOperations.h":
    void write_weights(const char* filename, PyObject* weights_list)
    PyObject* read_weight(const char* filename, unsigned long long weight_index)

# Python wrapper function to write weights
def write_weights_to_file(str filename, list weights_list):
    write_weights(filename.encode('utf-8'), <PyObject*> weights_list)

# Python wrapper function to read a weight
def load_weight_from_file(str filename, int weight_index):
    return <object> read_weight(filename.encode('utf-8'), weight_index)
