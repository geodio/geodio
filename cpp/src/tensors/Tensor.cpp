
#include <stdexcept>
#include <algorithm>

#include "Tensor.h"

namespace dio {

    std::vector<size_t> compute_strides_for_shape(const std::vector<size_t>& shape) {
        std::vector<size_t> strides(shape.size());
        if (shape.empty()) {
            return {};  // No strides for empty shape
        }
        strides.back() = 1;
        for (int i = static_cast<int>(shape.size()) - 2; i >= 0; --i) {
            strides[i] = strides[i + 1] * shape[i + 1];
        }
        return strides;
    }

    std::vector<size_t> compute_broadcast_strides(const std::vector<size_t>& original_strides,
                                                  const std::vector<size_t>& original_shape,
                                                  const std::vector<size_t>& broadcasted_shape) {
        size_t ndim1 = original_shape.size();
        size_t ndim2 = broadcasted_shape.size();
        size_t ndim = std::max(ndim1, ndim2);

        std::vector<size_t> new_strides(ndim, 0);

        for (size_t i = 0; i < ndim; ++i) {
            size_t orig_dim_idx = ndim1 > ndim2 ? i : i - (ndim - ndim1);
            size_t bcast_dim_idx = i - (ndim - ndim2);

            size_t orig_dim = (i < ndim - ndim1) ? 1 : original_shape[orig_dim_idx];
            size_t bcast_dim = (i < ndim - ndim2) ? 1 : broadcasted_shape[bcast_dim_idx];

            // Special case for 1D vector (p) to (p, 1)
            if (ndim1 == 1 && ndim2 == 2 && original_shape[0] == broadcasted_shape[0]) {
                new_strides[0] = original_strides[0]; // Use original stride for the first dimension
                new_strides[1] = 0;                   // Set stride 0 for the broadcasted second dimension
            }
            // Special case for (m) to (1, m)
            else if (ndim1 == 1 && ndim2 == 2 && original_shape[0] == broadcasted_shape[1]) {
                new_strides[0] = 0;                   // First dimension is broadcasted (set stride 0)
                new_strides[1] = original_strides[0]; // Use original stride for the second dimension
            }
            // General case for (m, 1) to (m, n) and other broadcastings
            else if (orig_dim == 1 && bcast_dim > 1) {
                new_strides[i] = 0; // Dimension is broadcasted, set stride to 0
            }
            // Stride preservation when dimensions match
            else if (orig_dim == bcast_dim) {
                new_strides[i] = original_strides[orig_dim_idx]; // If dimensions match, preserve stride
            }
            // Handle mismatched dimensions that are not broadcastable
            else if (orig_dim != bcast_dim && orig_dim > 1) {
                throw std::invalid_argument("Shapes are not broadcastable.");
            }
        }

        return new_strides;
    }


    template<typename T>
    [[nodiscard]] Tensor<T> Tensor<T>::slice(const std::vector<Slice>& slices) {
        // Instead of copying, we will return a tensor view
        auto tensor = Tensor<T>(*this, slices);
        return tensor;  // No copy, return a view
    }

    // Constructor for slicing (creates a view instead of a copy)
    template<typename T>
    Tensor<T>::Tensor(Tensor<T>& base_tensor, const std::vector<Slice>& slices)
        : base_tensor_(std::ref(base_tensor)), slices_(slices), is_view_(true), total_size_(base_tensor.total_size_) {
        calculate_view();  // Adjust shape, strides, etc. for the view
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

    // Helper function to compute flat index from multi-dimensional indices. It does not add the offset to the index!
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
    [[nodiscard]] std::vector<T> Tensor<T>::get_data() const {
        return this->is_view_? this->base_tensor_->get().data_: this->data_;
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

        // Create strides for the original tensor
        std::vector<size_t> new_strides(this->shape_.size());
        for (size_t i = 0; i < new_axis.size(); ++i) {
            new_strides[i] = this->strides_[new_axis[i]];
        }

        // Copy the data according to the new axis order
        for (size_t i = 0; i < result.total_size_; ++i) {
            // Compute the indices in the original shape
            std::vector<size_t> result_indices = result.compute_indices(i, result.shape_);

            // Permute the indices back to the original tensor's order using the new_axis
            std::vector<size_t> old_indices(result_indices.size());
            for (size_t j = 0; j < result_indices.size(); ++j) {
                old_indices[new_axis[j]] = result_indices[j];
            }

            // Compute the flat index in the original tensor based on the permuted indices
            size_t new_flat_index = compute_flat_index(old_indices, this->strides_) + (is_view_ ? offset_ : 0);

            // Assign the value to the transposed tensor
            result.data_[i] = is_view_ ? base_tensor_->get().data_[new_flat_index] : this->data_[new_flat_index];
        }

        return result;
    }


    template<typename T>
    Tensor<T> Tensor<T>::sum(const std::vector<size_t>& axis) const {
        if (axis.empty() || (axis.size() == 1 && axis[0] == 0)) {
            // Sum all elements (flatten the tensor)
            const T* data_ptr = is_view_ ? base_tensor_->get().data_.data() + offset_ : this->data_.data();
            T total = std::accumulate(data_ptr, data_ptr + this->total_size_, static_cast<T>(0));
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
                for (size_t j = 0; j < this->shape_[ax]; ++j) {
                    std::vector<size_t> varying_indices = indices;
                    varying_indices[ax] = j;

                    // Adjust the index calculation to account for views
                    size_t flat_index = compute_flat_index(varying_indices, this->strides_) + (is_view_ ? offset_ : 0);
                    sum_value += is_view_ ? base_tensor_->get().data_[flat_index] : this->data_[flat_index];
                }
            }

            result.data_[i] = sum_value;
        }

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



    template<typename T>
    bool TensorIterator<T>::operator==(const TensorIterator &other) const {
        return current_index_ == other.current_index_;
    }

    template<typename T>
    bool TensorIterator<T>::operator!=(const TensorIterator &other) const {
        return !(*this == other);
    }

    template<typename T>
    TensorIterator<T> &TensorIterator<T>::operator++() {
        ++current_index_;
        return *this;
    }

    template<typename T>
    TensorIterator<T> TensorIterator<T>::operator++(int) {
        TensorIterator temp = *this;
        ++(*this);
        return temp;
    }

    template<typename T>
    const T& TensorIterator<T>::operator*() const {
        std::vector<size_t> indices = tensor_->compute_indices(current_index_, tensor_->shape());
        return (*tensor_)(indices);  // Const reference
    }

//    template<typename T>
//    T& TensorIterator<T>::operator*() {
//        std::vector<size_t> indices = tensor_->compute_indices(current_index_, tensor_->shape());
//        return (*tensor_)(indices);  // Non-const reference
//    }

    // Explicit template instantiation (required in a separate compilation unit)
    template class TensorIterator<float>;
    template class TensorIterator<double>;
    template class TensorIterator<int>;
    template class TensorIterator<bool>;
    template class Tensor<float>;
    template class Tensor<double>;
    template class Tensor<int>;

} // namespace dio
