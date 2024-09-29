#include "FileOperations.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Function to write weights to the custom format
void write_weights(const char* filename, PyObject* weights_list) {
    FILE *file = fopen(filename, "wb");
    if (!file) {
        perror("Failed to open file");
        return;
    }

    // Initialize the numpy API
    import_array();

    // Get number of weights
    Py_ssize_t num_weights = PyList_Size(weights_list);

    // Allocate memory for offsets
    uint64_t *offsets = (uint64_t*)malloc(num_weights * sizeof(uint64_t));
    if (!offsets) {
        perror("Failed to allocate memory for offsets");
        fclose(file);
        return;
    }

    // Calculate offsets for each weight and track the current offset
    uint64_t current_offset = sizeof(uint64_t) * (1 + num_weights); // Start after the file header
    for (Py_ssize_t i = 0; i < num_weights; ++i) {
        PyArrayObject *array = (PyArrayObject*)PyList_GetItem(weights_list, i);
        offsets[i] = current_offset;
        current_offset += sizeof(uint64_t) +              // dimensionality
                         sizeof(uint64_t) * PyArray_NDIM(array) + // shape array
                         sizeof(uint64_t) +              // data size
                         PyArray_NBYTES(array);          // actual data size
    }

    // Write file header
    fwrite(&num_weights, sizeof(uint64_t), 1, file);        // Number of weights
    fwrite(offsets, sizeof(uint64_t), num_weights, file);   // Offsets of each weight

    // Write each weight with its header
    for (Py_ssize_t i = 0; i < num_weights; ++i) {
        PyArrayObject *array = (PyArrayObject*)PyList_GetItem(weights_list, i);
        uint64_t dimensionality = PyArray_NDIM(array);
        npy_intp *shape = PyArray_SHAPE(array);
        uint64_t data_size = PyArray_NBYTES(array);

        // Write weight header
        fwrite(&dimensionality, sizeof(uint64_t), 1, file);    // Dimensionality of the array
        fwrite(shape, sizeof(uint64_t), dimensionality, file); // Shape of the array
        fwrite(&data_size, sizeof(uint64_t), 1, file);         // Size of the data

        // Write the actual weight data
        fwrite(PyArray_DATA(array), 1, data_size, file);
    }

    // Cleanup
    free(offsets);
    fclose(file);
}

// Function to read a specific weight from the file
PyObject* read_weight(const char* filename, uint64_t weight_index) {
    FILE *file = fopen(filename, "rb");
    if (!file) {
        perror("Failed to open file");
        return NULL;
    }

    // Read the number of weights from the file header
    uint64_t num_weights;
    fread(&num_weights, sizeof(uint64_t), 1, file);

    // Verify the requested weight index is within bounds
    if (weight_index >= num_weights) {
        fclose(file);
        PyErr_SetString(PyExc_IndexError, "weight_index out of bounds");
        return NULL;
    }

    // Read offsets
    uint64_t *offsets = (uint64_t*)malloc(num_weights * sizeof(uint64_t));
    if (!offsets) {
        fclose(file);
        perror("Failed to allocate memory for offsets");
        return NULL;
    }
    fread(offsets, sizeof(uint64_t), num_weights, file);

    // Seek to the position of the requested weight
    fseek(file, offsets[weight_index], SEEK_SET);

    // Read weight header
    WeightHeader weight_header;
    fread(&weight_header.dimensionality, sizeof(uint64_t), 1, file);
    weight_header.shape = (uint64_t*)malloc(weight_header.dimensionality * sizeof(uint64_t));
    if (!weight_header.shape) {
        free(offsets);
        fclose(file);
        perror("Failed to allocate memory for weight shape");
        return NULL;
    }
    fread(weight_header.shape, sizeof(uint64_t), weight_header.dimensionality, file);
    fread(&weight_header.data_size, sizeof(uint64_t), 1, file);

    // Create a new numpy array to hold the weight
    npy_intp dims[weight_header.dimensionality];
    for (uint64_t i = 0; i < weight_header.dimensionality; ++i) {
        dims[i] = (npy_intp)weight_header.shape[i];
    }
    PyObject *array = PyArray_SimpleNew(weight_header.dimensionality, dims, NPY_FLOAT64);
    if (!array) {
        free(weight_header.shape);
        free(offsets);
        fclose(file);
        return NULL; // Error in creating numpy array
    }

    // Read the weight data into the numpy array
    fread(PyArray_DATA((PyArrayObject*)array), 1, weight_header.data_size, file);

    // Cleanup
    free(weight_header.shape);
    free(offsets);
    fclose(file);

    return array;
}
