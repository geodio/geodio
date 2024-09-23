#ifndef FILE_OPERATIONS_H
#define FILE_OPERATIONS_H

#include <stdint.h>
#include <Python.h>
#include <numpy/arrayobject.h>

// Structs for file and weight headers
typedef struct {
    uint64_t num_weights;     // Number of weights
    uint64_t max_offset_size; // Maximum byte size of the offsets
    uint64_t *offsets;        // Array of byte offsets for each weight
} FileHeader;

typedef struct {
    uint64_t dimensionality;  // Dimensionality of the array
    uint64_t *shape;          // Shape of the array
    uint64_t data_size;       // Size of the data in bytes
} WeightHeader;

// Function declarations
void write_weights(const char* filename, PyObject* weights_list);
PyObject* read_weight(const char* filename, uint64_t weight_index);

#endif // FILE_OPERATIONS_H
