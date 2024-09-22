
#include <stdexcept>
#include <algorithm>

#include "Tensor.h"



namespace dio {

template<typename T>
[[nodiscard]] Tensor<T> Tensor<T>::slice(const std::vector<Slice> &slices) {
    auto tensor = Tensor<T>(*this, slices);
    return tensor.copy();
}

// Constructor for slicing
template<typename T>
Tensor<T>::Tensor(Tensor<T>& base_tensor, const std::vector<Slice>& slices)
    : base_tensor_(std::ref(base_tensor)), slices_(slices), is_view_(true), total_size_(base_tensor.total_size_){
    calculate_view();
}

template<typename T>
Tensor<T>::Tensor()
    : total_size_(0), is_view_(false) {}

template<typename T>
Tensor<T>::Tensor(const T& value)
    : shape_(), total_size_(1), is_view_(false) {
    data_.resize(1);
    data_[0] = value;
    strides_ = {};
}

template<typename T>
Tensor<T>::Tensor(const std::vector<T>& data, const std::vector<size_t>& shape)
    : shape_(shape), data_(data), is_view_(false) {
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



template<typename T>
Tensor<T> Tensor<T>::transpose(const std::vector<size_t>& axis) const {
    Tensor<T> result;

    // If axis is empty or {0}, reverse the shape (default transpose)
    std::vector<size_t> new_axis = axis;
    if (new_axis.empty() || (new_axis.size() == 1 && new_axis[0] == 0)) {
        new_axis.resize(this->shape_.size());
        std::iota(new_axis.begin(), new_axis.end(), 0);  // Create [0, 1, 2, ..., N]
        std::reverse(new_axis.begin(), new_axis.end());  // Reverse it
    }

    // Ensure the new axis order is valid
    if (new_axis.size() != this->shape_.size()) {
        throw std::invalid_argument("Axis size must match tensor dimensions.");
    }

    // Set new shape based on the new axis order
    result.shape_.resize(this->shape_.size());
    for (size_t i = 0; i < new_axis.size(); ++i) {
        result.shape_[i] = this->shape_[new_axis[i]];
    }

    result.total_size_ = this->total_size_;
    result.data_.resize(this->total_size_);
    result.compute_strides();

    // Create strides for both tensors
    std::vector<size_t> new_strides(this->shape_.size());
    for (size_t i = 0; i < new_axis.size(); ++i) {
        new_strides[i] = this->strides_[new_axis[i]];
    }

    // Copy the data according to the new axis order
    for (size_t i = 0; i < result.total_size_; ++i) {
        std::vector<size_t> old_indices = compute_indices(i, result.shape_);
        size_t new_flat_index = compute_flat_index(old_indices, new_strides);
        result.data_[i] = this->data_[new_flat_index];
    }

    return result;
}

template<typename T>
Tensor<T> Tensor<T>::sum(const std::vector<size_t>& axis) const {
    if (axis.empty() || (axis.size() == 1 && axis[0] == 0)) {
        // Sum all elements (flatten the tensor)
        T total = std::accumulate(this->data_.begin(), this->data_.end(), static_cast<T>(0));
        return Tensor<T>(std::vector<T>{total}, {}); // Return scalar tensor
    }

    // Ensure axis dimensions are valid
    for (size_t ax : axis) {
        if (ax >= this->shape_.size()) {
            throw std::invalid_argument("Axis is out of bounds for tensor dimensions.");
        }
    }

    // Reduce along the provided axes
    std::vector<size_t> new_shape = this->shape_;
    for (size_t ax : axis) {
        new_shape[ax] = 1;
    }

    Tensor<T> result;
    result.shape_ = new_shape;
    result.total_size_ = compute_size(new_shape);
    result.data_.resize(result.total_size_);
    result.compute_strides();

    // Perform the summation along the specified axes
    for (size_t i = 0; i < result.total_size_; ++i) {
        std::vector<size_t> indices = compute_indices(i, result.shape_);
        T sum_value = 0;

        // Sum along the specified axes
        for (size_t ax : axis) {
            std::vector<size_t> varying_indices = indices;
            for (size_t j = 0; j < this->shape_[ax]; ++j) {
                varying_indices[ax] = j;
                sum_value += this->operator()(varying_indices);  // Access via operator()
            }
        }

        result.data_[i] = sum_value;
    }

    return result;
}



// Operators




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
