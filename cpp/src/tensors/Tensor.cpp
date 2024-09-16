#include "Tensor.h"
#include <fstream>
#include <numeric>
#include <stdexcept>
#include <algorithm>

namespace dio {
// Constructor: initialize from initializer list
template<typename T>
Tensor<T>::Tensor(std::initializer_list<T> list) {
    initialize_from_list(list);
}

// Constructor: initialize from std::vector and shape
template<typename T>
Tensor<T>::Tensor(const std::vector<T>& data, const std::vector<size_t>& shape)
    : data_(data), shape_(shape) {
    compute_strides();
}

// Constructor: initialize from raw pointer and shape
template<typename T>
Tensor<T>::Tensor(const T* data, const std::vector<size_t>& shape)
    : shape_(shape) {
    size_t total_size = compute_size(shape);
    data_.assign(data, data + total_size);
    compute_strides();
}

// Return number of dimensions
template<typename T>
size_t Tensor<T>::num_dimensions() const {
    return shape_.size();
}

// Return the shape
template<typename T>
const std::vector<size_t>& Tensor<T>::shape() const {
    return shape_;
}

// Broadcasting logic
template<typename T>
void Tensor<T>::broadcast(const std::vector<size_t>& new_shape) {
    if (new_shape.size() < shape_.size()) {
        throw std::invalid_argument("New shape has fewer dimensions than current shape.");
    }

    std::vector<size_t> new_shape_temp = shape_;
    new_shape_temp.insert(new_shape_temp.begin(), new_shape.size() - shape_.size(), 1);

    for (size_t i = 0; i < shape_.size(); ++i) {
        if (new_shape[i] != new_shape_temp[i] && new_shape_temp[i] != 1) {
            throw std::invalid_argument("Broadcasting incompatible shapes.");
        }
    }

    shape_ = new_shape;
    compute_strides();
}

// Compute strides based on shape
template<typename T>
void Tensor<T>::compute_strides() {
    strides_.resize(shape_.size());
    size_t stride = 1;
    for (int i = shape_.size() - 1; i >= 0; --i) {
        strides_[i] = stride;
        stride *= shape_[i];
    }
}

// Save tensor to file
template<typename T>
void Tensor<T>::save_to_file(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing.");
    }

    size_t shape_size = shape_.size();
    file.write(reinterpret_cast<const char*>(&shape_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(shape_.data()), sizeof(size_t) * shape_size);

    size_t data_size = data_.size();
    file.write(reinterpret_cast<const char*>(&data_size), sizeof(size_t));
    file.write(reinterpret_cast<const char*>(data_.data()), sizeof(T) * data_size);

    file.close();
}

// Load tensor from file
template<typename T>
void Tensor<T>::load_from_file(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading.");
    }

    size_t shape_size;
    file.read(reinterpret_cast<char*>(&shape_size), sizeof(size_t));
    shape_.resize(shape_size);
    file.read(reinterpret_cast<char*>(shape_.data()), sizeof(size_t) * shape_size);

    size_t data_size;
    file.read(reinterpret_cast<char*>(&data_size), sizeof(size_t));
    data_.resize(data_size);
    file.read(reinterpret_cast<char*>(data_.data()), sizeof(T) * data_size);

    compute_strides();
    file.close();
}

// Element access (multi-dimensional indexing)
template<typename T>
T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides_[i];
    }
    return data_[index];
}

template<typename T>
const T& Tensor<T>::operator()(const std::vector<size_t>& indices) const {
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides_[i];
    }
    return data_[index];
}

// Element-wise addition
template<typename T>
Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have the same shape for element-wise addition.");
    }

    Tensor<T> result = *this;
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] + other.data_[i];
    }
    return result;
}

// Element-wise multiplication
template<typename T>
Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
    if (shape_ != other.shape_) {
        throw std::invalid_argument("Tensors must have the same shape for element-wise multiplication.");
    }

    Tensor<T> result = *this;
    for (size_t i = 0; i < data_.size(); ++i) {
        result.data_[i] = data_[i] * other.data_[i];
    }
    return result;
}

// GPU Integration Placeholder (to be implemented)
template<typename T>
void Tensor<T>::to_gpu() {
    // GPU data transfer logic (to be implemented)
}

template<typename T>
void Tensor<T>::from_gpu() {
    // GPU data transfer logic (to be implemented)
}

template<typename T>
void Tensor<T>::initialize_from_list(const std::initializer_list<T>& list) {
    shape_.push_back(list.size());
    data_.assign(list.begin(), list.end());
    compute_strides();
}

// Helper function to calculate the total size of tensor
template<typename T>
size_t Tensor<T>::compute_size(const std::vector<size_t>& shape) const {
    return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<>());
}

// Explicit template instantiation (required in a separate compilation unit)
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;
} // dio