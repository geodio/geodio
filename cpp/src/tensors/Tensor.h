// Tensor.h

#ifndef GEODIO_TENSOR_H
#define GEODIO_TENSOR_H

#include "../backends/Backend.h"
#include "../backends/CPUBackend.h"
#include "../backends/BackendManager.h"

#include "ITensor.h"
#include "Slice.h"
#include <initializer_list>
#include <vector>
#include <optional>
#include <cstddef>
#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <type_traits>
#include <numeric>
#include <stdexcept>
#include <algorithm>
#include <memory>
#include <functional>

namespace dio {

    std::vector<size_t> compute_strides_for_shape(const std::vector<size_t>& shape);

    std::vector<size_t> compute_broadcast_strides(
        const std::vector<size_t>& original_strides,
        const std::vector<size_t>& original_shape,
        const std::vector<size_t>& broadcasted_shape);

    template<typename T>
    class Tensor : public ITensor{
    public:
        // Constructors
        Tensor();  // Default constructor

        // Constructor from vector and shape
        Tensor(const std::vector<T>& data, const std::vector<size_t>& shape);

        // Constructor for scalar tensors
        explicit Tensor(const T& value);

        Tensor(Tensor<T> &base_tensor, const std::vector<Slice> &slices);

        // Destructor
        ~Tensor() override;

        [[nodiscard]] const std::type_info& type() const {
            return typeid(T);
        }
        [[nodiscard]] const std::type_info& type_info() const override {
            return typeid(T);
        }



        // Information methods
        [[nodiscard]] size_t num_dimensions() const;
        [[nodiscard]] const std::vector<size_t>& shape() const;
        [[nodiscard]] const std::vector<size_t>& strides() const; // Public accessor for strides
        [[nodiscard]] size_t compute_size() const;
        [[nodiscard]] size_t compute_size(const std::vector<size_t>& shape) const;
        [[nodiscard]] bool is_scalar() const;

        std::vector<T> get_data() const;
        // Element access
        T& operator()(const std::vector<size_t>& indices);

        const T& operator()(const std::vector<size_t>& indices) const;

        Tensor<T> slice(const std::vector<Slice>& slices);


        // Overload operator= for tensor assignment
        Tensor<T>& operator=(const std::vector<T>& values) {
            if (values.size() != data_.size()) {
                throw std::invalid_argument("Assignment size mismatch");
            }
            std::copy(values.begin(), values.end(), data_.begin());
            return *this;
        }

        // Overload operator= for scalar assignment
        Tensor<T>& operator=(const T& scalar) {
            std::fill(data_.begin(), data_.end(), scalar);
            return *this;
        }

        // Element-wise operations
        template<typename U, typename R>
        Tensor<R> elementwise_binary_operation(const Tensor<U>& other, std::function<R(T, U)> func) const;


        template<typename U>
        Tensor<typename std::common_type<T, U>::type> add(const Tensor<U>& other) const;

        template<typename U>
        Tensor<typename std::common_type<T, U>::type> subtract(const Tensor<U>& other) const;

        template<typename U>
        Tensor<typename std::common_type<T, U>::type> multiply(const Tensor<U>& other) const;

        template<typename U>
        Tensor<typename std::common_type<T, U>::type> divide(const Tensor<U>& other) const;

        // Element-wise functions
        template<typename R>
        Tensor<R> apply_elementwise_function(std::function<R(T)> func) const;

        // Matrix multiplication
        template<typename U>
        Tensor<typename std::common_type<T, U>::type> matmul(const Tensor<U>& other) const;

        // Operators
        template<typename U>
        Tensor<typename std::common_type<T, U>::type> operator+(const Tensor<U>& other) const;

        template<typename U>
        Tensor<typename std::common_type<T, U>::type> operator-(const Tensor<U>& other) const;

        template<typename U>
        Tensor<typename std::common_type<T, U>::type> operator*(const Tensor<U>& other) const;

        template<typename U>
        Tensor<typename std::common_type<T, U>::type> operator/(const Tensor<U>& other) const;

        // To string representation
        [[nodiscard]] std::string to_string() const;

        // Friend operator<< for printing
        template<typename U>
        friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);

        // GPU operations
        void to_device();
        void from_device();

        template<typename U>
        Tensor<U> cast() const {

            // If T and U are the same type, return a copy of the tensor
            if (std::is_same<U, T>::value) {
                return *this;  // Implicitly converts Tensor<T> to Tensor<U> when T == U
            }
            // Otherwise, create a new Tensor<U> and convert the data
            Tensor<U> result;
            result.shape_ = this->shape_;
            result.total_size_ = this->total_size_;
            result.compute_strides(); // Ensure this correctly initializes strides_

            // Resize the data vector to hold the new type
            result.data_.resize(this->total_size_);

            // Perform the type conversion
            std::transform(
                data_.begin(),
                data_.end(),
                result.data_.begin(),
                [](const T& value) {
                    return static_cast<U>(value);
                }
            );
            return result;

        }

        Tensor <T> transpose(const std::vector<size_t> &axis) const;

        Tensor <T> sum(const std::vector<size_t> &axis) const;

        Tensor<T> copy() const;

        // private:
        std::vector<T> data_;           // Changed from raw pointer to std::vector

        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t total_size_;

        bool is_view_;                  // Flag to indicate if this tensor is a view (slice)

        // For slicing (view)
        size_t offset_ = 0;
        std::optional<std::reference_wrapper<Tensor<T>>> base_tensor_;

        std::vector<Slice> slices_ {};  // Slice info for each dimension

        // Helper methods
        void compute_strides();

        [[nodiscard]] std::vector<size_t> compute_broadcast_shape(
        const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) const;

        [[nodiscard]] size_t compute_flat_index(
        const std::vector<size_t>& indices, const std::vector<size_t>& strides) const;
        [[nodiscard]] std::vector<size_t> compute_indices(
        size_t flat_index, const std::vector<size_t>& shape) const;
        // Slice handling
        void calculate_view();

        // Calculate base tensor indices from the view's indices
        [[nodiscard]] std::vector<size_t> calculate_base_indices(const std::vector<size_t>& view_indices) const;

        // Increment view indices for traversing the view (used for copying view data)
        void increment_indices(std::vector<size_t>& indices) const;

        // Helper function to print tensor contents
        void print_tensor(std::ostream& os, size_t index, size_t depth) const;

        template<typename U>
        void broadcast_shapes(const Tensor<U>& other,
        std::vector<size_t>& out_shape,
        std::vector<size_t>& adjusted_strides1,
        std::vector<size_t>& adjusted_strides2) const;
    };

    template<typename T>
    void Tensor<T>::increment_indices(std::vector<size_t> &indices) const {
        size_t i;
        for (i = indices.size(); i > 0; --i) {
            indices[i - 1]++;
            if (indices[i - 1] < shape_[i - 1]) {
                break;
            } else if (i > 1) {
                indices[i - 1] = 0;
            }
        }
    }

    template<typename T>
    Tensor<T> Tensor<T>::copy() const {
        if (is_view_) {
            // Create a copy from the view
            std::vector<T> view_data;
            view_data.reserve(total_size_);  // Reserve space for the view's data

            // Extract the data from the base tensor using the slice indices
            std::vector<size_t> view_indices(shape_.size(), 0);
            for (size_t i = 0; i < total_size_; ++i) {
                view_data.push_back((*this)(view_indices));
                increment_indices(view_indices);  // Move to the next index in the view
            }

            // Create and return a new full tensor from the view's data
            return Tensor<T>(view_data, shape_);
        } else {
            // Simple hard copy for the full tensor
            return Tensor<T>(data_, shape_);
        }
    }

    template<typename T>
    std::vector<size_t> Tensor<T>::calculate_base_indices(const std::vector<size_t>& view_indices) const {
        std::vector<size_t> base_indices(view_indices.size());

        for (size_t i = 0; i < view_indices.size(); ++i) {
            const Slice& slice = slices_[i];
            int start = slice.resolve_start(base_tensor_->get().shape()[i]);
            base_indices[i] = start + view_indices[i] * slice.step();
        }
        return base_indices;
    }

    template<typename T>
    void Tensor<T>::calculate_view() {
        shape_.clear();
        strides_.clear();

        size_t num_dimensions = base_tensor_->get().shape().size();
        offset_ = 0;  // Reset the offset

        // Handle case where fewer slices are provided than there are dimensions
        for (size_t i = 0; i < num_dimensions; ++i) {
            if (i < slices_.size()) {
                const auto& slice = slices_[i];

                // Resolve start and end indices
                int start = slice.resolve_start(base_tensor_->get().shape()[i]);
                int end = slice.resolve_end(base_tensor_->get().shape()[i]);

                // Adjust for negative indices AFTER knowing the shape
                if (start < 0) {
                    start += base_tensor_->get().shape()[i];
                }
                if (end < 0) {
                    end += base_tensor_->get().shape()[i];
                }

                // Check for out-of-bounds indices
                if (start < 0 || end < 0 || start >= base_tensor_->get().shape()[i] || end > base_tensor_->get().shape()[i]) {
                    throw std::out_of_range("Slice index out of range");
                }

                // Calculate the offset in the flattened array
                offset_ += start * base_tensor_->get().strides()[i];

                // Set the new shape and strides for the view
                shape_.push_back(end - start);
                strides_.push_back(base_tensor_->get().strides()[i] * slice.step());
            } else {
                // If no slice is provided for this dimension, treat it as an empty slice (full range)
                shape_.push_back(base_tensor_->get().shape()[i]);
                strides_.push_back(base_tensor_->get().strides()[i]);
            }
        }

        // Note: No need to adjust data_ directly. The data pointer remains empty in the view.
    }

    // Implementation of methods (included in header due to templates)
    // Element access (multi-dimensional indexing)
    template<typename T>
    T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
        std::vector<size_t> expanded_indices = indices;

        // Automatically handle empty slices by filling missing dimensions with 0
        while (expanded_indices.size() < shape_.size()) {
            expanded_indices.push_back(0);
        }

        if (expanded_indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices must match the number of dimensions.");
        }

        for (size_t i = 0; i < expanded_indices.size(); ++i) {
            if (expanded_indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
        }

        size_t index = compute_flat_index(expanded_indices, strides_);
        return is_view_ ? base_tensor_->get().data_[index + offset_] : data_[index];
    }

    template<typename T>
    const T& Tensor<T>::operator()(const std::vector<size_t>& indices) const {
        std::vector<size_t> expanded_indices = indices;

        // Automatically handle empty slices by filling missing dimensions with 0
        while (expanded_indices.size() < shape_.size()) {
            expanded_indices.push_back(0);
        }

        if (expanded_indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices must match the number of dimensions.");
        }

        for (size_t i = 0; i < expanded_indices.size(); ++i) {
            if (expanded_indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
        }

        size_t index = compute_flat_index(expanded_indices, strides_);
        return is_view_ ? base_tensor_->get().data_[index + offset_] : data_[index];
    }

    // To string representation
    template<typename T>
    std::string Tensor<T>::to_string() const {
        std::ostringstream ss;
        ss << *this;
        return ss.str();
    }

    // Overload of the << operator to print tensor contents
    template<typename U>
    std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor) {
        if (tensor.num_dimensions() == 0) {
            // Scalar tensor
            os << tensor.data_[0];
        } else {
            tensor.print_tensor(os, 0, 0);
        }
        return os;
    }

    // Helper function to print tensor contents recursively, handling both regular and view tensors
    template<typename T>
    void Tensor<T>::print_tensor(std::ostream& os, size_t index, size_t depth) const {
        // Check if the tensor is a view and adjust the data access accordingly
        const T* data_ptr;
        if (is_view_) {
            // Retrieve data from the base tensor if this is a view
            data_ptr = base_tensor_->get().data_.data() + offset_;
        } else {
            // Use the local data if this is not a view
            data_ptr = data_.data();
        }

        if (depth == shape_.size() - 1) {
            os << "[";
            for (size_t i = 0; i < shape_[depth]; ++i) {
                os << data_ptr[index + i];  // Access the correct data pointer
                if (i < shape_[depth] - 1) {
                    os << ", ";
                }
            }
            os << "]";
        } else {
            os << "[";
            for (size_t i = 0; i < shape_[depth]; ++i) {
                size_t new_index = index + i * strides_[depth];
                print_tensor(os, new_index, depth + 1);  // Recursive call for deeper dimensions
                if (i < shape_[depth] - 1) {
                    os << ", ";
                }
            }
            os << "]";
        }
    }

    // Element-wise binary operation
    template<typename T>
    template<typename U, typename R>
    Tensor<R> Tensor<T>::elementwise_binary_operation(const Tensor<U>& other, std::function<R(T, U)> func) const {
        // Compute broadcasted shape
        std::vector<size_t> result_shape;
        std::vector<size_t> adjusted_strides1, adjusted_strides2;
        broadcast_shapes(other, result_shape, adjusted_strides1, adjusted_strides2);

        size_t total_size = compute_size(result_shape);
        Tensor<R> result;
        result.shape_ = result_shape;
        result.total_size_ = total_size;
        result.data_.resize(total_size);
        result.compute_strides();

        // Prepare adjusted strides and data pointers
        const T* data1 = is_view_ ? base_tensor_->get().data_.data() + offset_ : this->data_.data();
        const U* data2 = other.is_view_ ? other.base_tensor_->get().data_.data() + other.offset_ : other.data_.data();
        R* result_data = result.data_.data();

        auto backend = BackendManager<T>::get_backend();

        // Use the backend's elementwise operation
        backend->template elementwise_operation<U, R>(
            data1, data2, result_data,
            func,
            total_size, result_shape, adjusted_strides1, adjusted_strides2
        );

        return result;
    }

    // Element-wise addition
    template<typename T>
    template<typename U>
    Tensor<typename std::common_type<T, U>::type> Tensor<T>::add(const Tensor<U>& other) const {
        using ResultType = typename std::common_type<T, U>::type;
        return this->template elementwise_binary_operation<U, ResultType>(other, [](T x, U y) { return x + y; });
    }

    // Subtraction
    template<typename T>
    template<typename U>
    Tensor<typename std::common_type<T, U>::type> Tensor<T>::subtract(const Tensor<U>& other) const {
        using ResultType = typename std::common_type<T, U>::type;
        return this->template elementwise_binary_operation<U, ResultType>(other, [](T x, U y) { return x - y; });
    }

    // Multiplication
    template<typename T>
    template<typename U>
    Tensor<typename std::common_type<T, U>::type> Tensor<T>::multiply(const Tensor<U>& other) const {
        using ResultType = typename std::common_type<T, U>::type;
        return this->template elementwise_binary_operation<U, ResultType>(other, [](T x, U y) { return x * y; });
    }

    // Division
    template<typename T>
    template<typename U>
    Tensor<typename std::common_type<T, U>::type> Tensor<T>::divide(const Tensor<U>& other) const {
        using ResultType = typename std::common_type<T, U>::type;
        return this->template elementwise_binary_operation<U, ResultType>(other, [](T x, U y) { return x / y; });
    }

    // Operators
    template<typename T>
    template<typename U>
    Tensor<typename std::common_type<T, U>::type> Tensor<T>::operator+(const Tensor<U>& other) const {
        return this->add(other);
    }

    template<typename T>
    template<typename U>
    Tensor<typename std::common_type<T, U>::type> Tensor<T>::operator-(const Tensor<U>& other) const {
        return this->subtract(other);
    }

    template<typename T>
    template<typename U>
    Tensor<typename std::common_type<T, U>::type> Tensor<T>::operator*(const Tensor<U>& other) const {
        return this->multiply(other);
    }

    template<typename T>
    template<typename U>
    Tensor<typename std::common_type<T, U>::type> Tensor<T>::operator/(const Tensor<U>& other) const {
        return this->divide(other);
    }

    // Matrix multiplication
    template<typename T>
    template<typename U>
    [[nodiscard]] Tensor<typename std::common_type<T, U>::type>
    Tensor<T>::matmul(const Tensor<U>& other) const {
        std::vector<size_t> shape1 = shape_;
        std::vector<size_t> shape2 = other.shape_;

        // For vector handling, add dimension if necessary (1D -> 2D)
        if (shape1.size() == 1) {
            shape1.insert(shape1.begin(), 1);  // Treat as row vector
        }
        if (shape2.size() == 1) {
            shape2.push_back(1);  // Treat as column vector
        }

        // Check if dimensions allow broadcasting
        size_t n = shape1.back();  // Shared dimension from (x, n)
        size_t p = shape2.front(); // Shared dimension from (p, y)

        // Error check for incompatible dimensions where broadcasting isn't possible
        if (n != p && n != 1 && p != 1) {
            throw std::invalid_argument("Matrix multiplication dimensions mismatch: "
                                        "Incompatible shapes " + std::to_string(n) + " and " + std::to_string(p));
        }

        // Broadcast smaller dimension if possible
        if (n == 1) {
            shape1.back() = p; // Broadcast first matrix's trailing dimension
        } else if (p == 1) {
            shape2.front() = n; // Broadcast second matrix's leading dimension
        }

        // Compute adjusted shapes and strides
        std::vector<size_t> shape_a = shape1;  // Adjusted shape of A after broadcasting
        std::vector<size_t> shape_b = shape2;  // Adjusted shape of B after broadcasting

        // Adjust strides to account for broadcasting
        std::vector<size_t> strides_a = compute_broadcast_strides(
            strides_, shape_, shape_a);

        std::vector<size_t> strides_b = compute_broadcast_strides(
            other.strides_, other.shape_, shape_b);

        // Compute result shape
        std::vector<size_t> result_shape = { shape_a[0], shape_b[1] }; // (m, k)
        size_t m = result_shape[0];  // Number of rows in the result
        size_t k = result_shape[1];  // Number of columns in the result
        size_t n_shared = shape_a[1]; // Shared inner dimension

        // Initialize result tensor
        using R = typename std::common_type<T, U>::type;
        Tensor<R> result;
        result.shape_ = result_shape;
        result.total_size_ = m * k;
        result.data_.resize(result.total_size_);
        result.compute_strides();

        // Prepare data pointers
        const T* data1 = is_view_ ? base_tensor_->get().data_.data() + offset_ : data_.data();
        const U* data2 = other.is_view_ ? other.base_tensor_->get().data_.data() + other.offset_ : other.data_.data();
        R* result_data = result.data_.data();

        auto backend = BackendManager<T>::get_backend();

        // Use the backend's matrix multiplication
        backend->template matmul<U, R>(
            data1, data2, result_data,
            m, n_shared, k,
            strides_a, strides_b
        );

        return result;
    }

    // Apply element-wise function
    template<typename T>
    template<typename R>
    Tensor<R> Tensor<T>::apply_elementwise_function(std::function<R(T)> func) const {
        Tensor<R> result;
        result.shape_ = this->shape_;
        result.total_size_ = this->total_size_;
        result.data_.resize(result.total_size_);
        result.compute_strides();

        const T* data_in = is_view_ ? base_tensor_->get().data_.data() + offset_ : this->data_.data();
        R* data_out = result.data_.data();

        auto backend = BackendManager<T>::get_backend();

        backend->apply_unary_function(data_in, data_out, func, result.total_size_);

        return result;
    }

    // Broadcast shapes and compute adjusted strides
    template<typename T>
    template<typename U>
    void Tensor<T>::broadcast_shapes(const Tensor<U>& other,
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
} // namespace dio

#endif // GEODIO_TENSOR_H
