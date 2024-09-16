#include "Tensor.h"
#include <stdexcept>
#include <algorithm>
#include "../backends/CPUBackend.h"

namespace dio {

    template<typename T>
    Tensor<T>::Tensor()
        : data_(nullptr), total_size_(0), backend_(std::make_shared<CPUBackend<T>>()) {}

    template<typename T>
    Tensor<T>::Tensor(const T& value)
        : shape_(), total_size_(1), backend_(std::make_shared<CPUBackend<T>>()) {
        backend_->allocate(data_, 1);
        data_[0] = value;
        strides_ = {};
    }

    template<typename T>
    Tensor<T>::Tensor(const std::vector<T>& data, const std::vector<size_t>& shape)
        : shape_(shape), backend_(std::make_shared<CPUBackend<T>>()) {
        total_size_ = compute_size(shape_);
        backend_->allocate(data_, total_size_);
        std::copy(data.begin(), data.end(), data_);
        compute_strides();
    }

    template<typename T>
    Tensor<T>::~Tensor() {
        if (data_) {
            backend_->deallocate(data_);
        }
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

    // Element-wise addition with broadcasting
    template<typename T>
    Tensor<T> Tensor<T>::operator+(const Tensor<T>& other) const {
        std::vector<size_t> result_shape = compute_broadcast_shape(this->shape_, other.shape_);
        size_t total_size = compute_size(result_shape);
        Tensor<T> result(std::vector<T>(total_size), result_shape);
        result.compute_strides();

        // Precompute strides for broadcasting
        std::vector<size_t> strides1 = this->strides_;
        std::vector<size_t> strides2 = other.strides_;
        strides1.insert(strides1.begin(), result_shape.size() - strides1.size(), this->strides_.front());
        strides2.insert(strides2.begin(), result_shape.size() - strides2.size(), other.strides_.front());

        // Adjust shapes for broadcasting
        std::vector<size_t> shape1 = this->shape_;
        std::vector<size_t> shape2 = other.shape_;
        shape1.insert(shape1.begin(), result_shape.size() - shape1.size(), 1);
        shape2.insert(shape2.begin(), result_shape.size() - shape2.size(), 1);

        #pragma omp parallel for
        for (size_t i = 0; i < total_size; ++i) {
            // Compute indices in result tensor
            std::vector<size_t> idx = compute_indices(i, result_shape);

            // Adjust indices for operands
            std::vector<size_t> idx1(idx.size()), idx2(idx.size());
            for (size_t j = 0; j < idx.size(); ++j) {
                idx1[j] = (shape1[j] == 1) ? 0 : idx[j];
                idx2[j] = (shape2[j] == 1) ? 0 : idx[j];
            }

            size_t flat_idx1 = compute_flat_index(idx1, strides1);
            size_t flat_idx2 = compute_flat_index(idx2, strides2);

            result.data_[i] = this->data_[flat_idx1] + other.data_[flat_idx2];
        }
        return result;
    }

    // Element-wise multiplication with broadcasting
    template<typename T>
    Tensor<T> Tensor<T>::operator*(const Tensor<T>& other) const {
        // Implementation similar to operator+, replace '+' with '*'
        // [Same code as operator+ but with multiplication]
        std::vector<size_t> result_shape = compute_broadcast_shape(this->shape_, other.shape_);
        size_t total_size = compute_size(result_shape);
        Tensor<T> result(std::vector<T>(total_size), result_shape);
        result.compute_strides();

        // Precompute strides for broadcasting
        std::vector<size_t> strides1 = this->strides_;
        std::vector<size_t> strides2 = other.strides_;
        strides1.insert(strides1.begin(), result_shape.size() - strides1.size(), this->strides_.front());
        strides2.insert(strides2.begin(), result_shape.size() - strides2.size(), other.strides_.front());

        // Adjust shapes for broadcasting
        std::vector<size_t> shape1 = this->shape_;
        std::vector<size_t> shape2 = other.shape_;
        shape1.insert(shape1.begin(), result_shape.size() - shape1.size(), 1);
        shape2.insert(shape2.begin(), result_shape.size() - shape2.size(), 1);

        #pragma omp parallel for
        for (size_t i = 0; i < total_size; ++i) {
            // Compute indices in result tensor
            std::vector<size_t> idx = compute_indices(i, result_shape);

            // Adjust indices for operands
            std::vector<size_t> idx1(idx.size()), idx2(idx.size());
            for (size_t j = 0; j < idx.size(); ++j) {
                idx1[j] = (shape1[j] == 1) ? 0 : idx[j];
                idx2[j] = (shape2[j] == 1) ? 0 : idx[j];
            }

            size_t flat_idx1 = compute_flat_index(idx1, strides1);
            size_t flat_idx2 = compute_flat_index(idx2, strides2);

            result.data_[i] = this->data_[flat_idx1] * other.data_[flat_idx2];
        }
        return result;
    }


    template<typename T>
    std::vector<T> Tensor<T>::get_data() const {
        std::vector<T> data(data_, data_ + total_size_);
        return data;
    }


    template<typename T>
    void Tensor<T>::to_device() {
        // If data is already on device, skip
        // For CPUBackend, this might be a no-op
    }

    template<typename T>
    void Tensor<T>::from_device() {
        // For CPUBackend, this might be a no-op
    }
// Explicit template instantiation (required in a separate compilation unit)
template class Tensor<float>;
template class Tensor<double>;
template class Tensor<int>;

} // namespace dio
