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
        std::vector<T> data_;          // Changed from raw pointer to std::vector

        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t total_size_;
        bool is_view_;                        // Flag to indicate if this tensor is a view (slice)
        // For slicing (view)
        std::optional<std::reference_wrapper<Tensor<T>>> base_tensor_;

        std::vector<Slice> slices_ {};           // Slice info for each dimension
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
            base_indices[i] = slice.start() + view_indices[i] * slice.step();
        }
        return base_indices;
    }

   template<typename T>
    void Tensor<T>::calculate_view() {
        // Ensure the shape and strides vectors are the same size as the base tensor's shape
        shape_.resize(base_tensor_->get().shape().size());
        strides_.resize(base_tensor_->get().strides_.size());

        // Loop through all dimensions of the base tensor
        for (size_t i = 0; i < base_tensor_->get().shape().size(); ++i) {
            if (i < slices_.size()) {
                // If there's a slice for this dimension, apply it
                const Slice& slice = slices_[i];
                shape_[i] = (slice.end() - slice.start() + slice.step() - 1) / slice.step();  // Handle rounding up
                strides_[i] = base_tensor_->get().strides()[i] * slice.step();
            } else {
                // If no slice is provided, keep the original dimension's size and stride
                shape_[i] = base_tensor_->get().shape()[i];
                strides_[i] = base_tensor_->get().strides()[i];
            }
        }

        // Calculate the total size for the view based on the updated shape
        total_size_ = compute_size(shape_);
    }


    // Implementation of methods (included in header due to templates)

    // Element access (multi-dimensional indexing)
    template<typename T>
    T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
        if (is_view_) {
            // Calculate base indices from slice view and access base tensor
            std::vector<size_t> base_indices = calculate_base_indices(indices);
            return base_tensor_->get().operator()(base_indices);
        }
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices must match number of dimensions.");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds in dimension " + std::to_string(i));
            }
        }
        size_t index = compute_flat_index(indices, strides_);
        return data_[index];
    }

    template<typename T>
    const T& Tensor<T>::operator()(const std::vector<size_t>& indices) const {
        if (is_view_) {
            // Calculate base indices from slice view and access base tensor
            const std::vector<size_t> base_indices = calculate_base_indices(indices);
            return base_tensor_->get().operator()(base_indices);
        }
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices must match number of dimensions.");
        }
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds in dimension " + std::to_string(i));
            }
        }
        size_t index = compute_flat_index(indices, strides_);
        return data_[index];
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

    // Helper function to print tensor contents recursively
    template<typename T>
    void Tensor<T>::print_tensor(std::ostream& os, size_t index, size_t depth) const {
        if (depth == shape_.size() - 1) {
            os << "[";
            for (size_t i = 0; i < shape_[depth]; ++i) {
                os << data_[index + i];
                if (i < shape_[depth] - 1) {
                    os << ", ";
                }
            }
            os << "]";
        } else {
            os << "[";
            for (size_t i = 0; i < shape_[depth]; ++i) {
                size_t new_index = index + i * strides_[depth];
                print_tensor(os, new_index, depth + 1);
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
        const T* data1 = this->data_.data();
        const U* data2 = other.data_.data();
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
    [[nodiscard]] Tensor<typename std::common_type<T, U>::type> Tensor<T>::matmul(const Tensor<U>& other) const {
        if (this->num_dimensions() != 2 || other.num_dimensions() != 2) {
            throw std::invalid_argument("Both tensors must be 2D for matrix multiplication.");
        }
        if (this->shape_[1] != other.shape_[0]) {
            throw std::invalid_argument("Inner dimensions must match for matrix multiplication.");
        }
        using R = typename std::common_type<T, U>::type;

        size_t m = this->shape_[0];
        size_t n = this->shape_[1];
        size_t k = other.shape_[1];

        Tensor<R> result;
        result.shape_ = { m, k };
        result.total_size_ = m * k;
        result.data_.resize(result.total_size_);
        auto backend = BackendManager<T>::get_backend();
        result.compute_strides();

        const T* data1 = this->data_.data();
        const U* data2 = other.data_.data();
        R* result_data = result.data_.data();

        backend->matmul(data1, data2, result_data, m, n, k);

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

        const T* data_in = this->data_.data();
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
