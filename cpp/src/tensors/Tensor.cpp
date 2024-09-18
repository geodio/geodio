#include "Tensor.h"
#include <stdexcept>
#include <algorithm>

namespace dio {

template<typename T>
Tensor<T>::Tensor()
    : total_size_(0) {}

template<typename T>
Tensor<T>::Tensor(const T& value)
    : shape_(), total_size_(1) {
    data_.resize(1);
    data_[0] = value;
    strides_ = {};
}

template<typename T>
Tensor<T>::Tensor(const std::vector<T>& data, const std::vector<size_t>& shape)
    : shape_(shape), data_(data) {
    total_size_ = compute_size(shape_);
    if (data_.size() != total_size_) {
        throw std::invalid_argument("Data size does not match shape size.");
    }
    compute_strides();
}

template<typename T>
Tensor<T>::~Tensor() = default;

// Accessor for strides_
template<typename T>
const std::vector<size_t>& Tensor<T>::strides() const {
    return strides_;
}


// Helper function to compute broadcasted shape
template<typename T>
std::vector<size_t> Tensor<T>::compute_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) const {
    std::vector<size_t> result_shape;
    auto it1 = shape1.rbegin();
    auto it2 = shape2.rbegin();
    while (it1 != shape1.rend() || it2 != shape2.rend()) {
        size_t dim1 = (it1 != shape1.rend()) ? *it1 : 1;
        size_t dim2 = (it2 != shape2.rend()) ? *it2 : 1;
        if (dim1 == dim2 || dim1 == 1 || dim2 == 1) {
            result_shape.push_back(std::max(dim1, dim2));
        } else {
            throw std::invalid_argument("Shapes are not broadcastable.");
        }
        if (it1 != shape1.rend()) ++it1;
        if (it2 != shape2.rend()) ++it2;
    }
    std::reverse(result_shape.begin(), result_shape.end());
    return result_shape;
}

// Helper function to compute flat index from multi-dimensional indices
template<typename T>
size_t Tensor<T>::compute_flat_index(const std::vector<size_t>& indices, const std::vector<size_t>& strides) const {
    size_t index = 0;
    for (size_t i = 0; i < indices.size(); ++i) {
        index += indices[i] * strides[i];
    }
    return index;
}

// Helper function to compute multi-dimensional indices from flat index
template<typename T>
std::vector<size_t> Tensor<T>::compute_indices(size_t flat_index, const std::vector<size_t>& shape) const {
    std::vector<size_t> indices(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        indices[i] = (flat_index / strides_[i]) % shape[i];
    }
    return indices;
}

// Compute total size from shape
template<typename T>
size_t Tensor<T>::compute_size() const {
    return compute_size(shape_);
}

// Compute total size from a given shape
template<typename T>
size_t Tensor<T>::compute_size(const std::vector<size_t>& shape) const {
    return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<>());
}

// Compute strides based on shape
template<typename T>
void Tensor<T>::compute_strides() {
    strides_.resize(shape_.size());
    if (shape_.empty()) {
        strides_ = {};
        return;
    }
    strides_.back() = 1;
    for (int i = static_cast<int>(shape_.size()) - 2; i >= 0; --i) {
        strides_[i] = strides_[i + 1] * shape_[i + 1];
    }
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

// Check if tensor is scalar
template<typename T>
bool Tensor<T>::is_scalar() const {
    return shape_.empty();
}

// Get data as vector
template<typename T>
std::vector<T> Tensor<T>::get_data() const {
    return this->data_;
}



// Broadcast shapes and compute adjusted strides
template<typename T>
void Tensor<T>::broadcast_shapes(const Tensor<T>& other,
                                 std::vector<size_t>& out_shape,
                                 std::vector<size_t>& adjusted_strides1,
                                 std::vector<size_t>& adjusted_strides2) const {
    size_t ndim1 = this->shape_.size();
    size_t ndim2 = other.shape_.size();
    size_t ndim = std::max(ndim1, ndim2);

    out_shape.resize(ndim);
    adjusted_strides1.resize(ndim);
    adjusted_strides2.resize(ndim);

    for (size_t i = 0; i < ndim; ++i) {
        size_t dim1 = (i < ndim - ndim1) ? 1 : this->shape_[i - (ndim - ndim1)];
        size_t dim2 = (i < ndim - ndim2) ? 1 : other.shape_[i - (ndim - ndim2)];

        if (dim1 != dim2 && dim1 != 1 && dim2 != 1) {
            throw std::invalid_argument("Shapes are not broadcastable.");
        }

        out_shape[i] = std::max(dim1, dim2);
        adjusted_strides1[i] = (dim1 == 1) ? 0 : this->strides_[(ndim1 > ndim2 ? i : i - (ndim - ndim1))];
        adjusted_strides2[i] = (dim2 == 1) ? 0 : other.strides_[(ndim2 > ndim1 ? i : i - (ndim - ndim2))];
    }
}

// Matrix multiplication
template<typename T>
[[nodiscard]] Tensor<T> Tensor<T>::matmul(const Tensor<T>& other) const {
    if (this->num_dimensions() != 2 || other.num_dimensions() != 2) {
        throw std::invalid_argument("Both tensors must be 2D for matrix multiplication.");
    }
    if (this->shape_[1] != other.shape_[0]) {
        throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
    }

    size_t m = this->shape_[0];
    size_t n = this->shape_[1];
    size_t k = other.shape_[1];

    Tensor<T> result;
    result.shape_ = { m, k };
    result.total_size_ = m * k;
    result.data_.resize(result.total_size_);
    auto backend = BackendManager<T>::get_backend();
    result.compute_strides();

    const T* data1 = this->data_.data();
    const T* data2 = other.data_.data();
    T* result_data = result.data_.data();

    backend->matmul(data1, data2, result_data, m, n, k);

    return result;
}

// Operators

// Apply element-wise function
template<typename T>
Tensor<T> Tensor<T>::apply_elementwise_function(std::function<T(T)> func) const {
    Tensor<T> result;
    result.shape_ = this->shape_;
    result.total_size_ = this->total_size_;
    result.data_.resize(result.total_size_);
    result.compute_strides();

    const T* data_in = this->data_.data();
    T* data_out = result.data_.data();

    auto backend = BackendManager<T>::get_backend();

    backend->apply_unary_function(data_in, data_out, func, result.total_size_);

    return result;
}


// GPU operations (No-ops for CPU backend)
template<typename T>
void Tensor<T>::to_device() {
    // No-op for CPUBackend
}

template<typename T>
void Tensor<T>::from_device() {
    // No-op for CPUBackend
}


// Explicit template instantiation (required in a separate compilation unit)
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;

} // namespace dio
