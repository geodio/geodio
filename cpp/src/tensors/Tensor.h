// Tensor.h

#ifndef GEODIO_TENSOR_H
#define GEODIO_TENSOR_H

#include "../backends/Backend.h"
#include "../backends/CPUBackend.h"
#include "../backends/BackendManager.h"
#include <initializer_list>
#include <vector>
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
    class Tensor {
    public:
        // Constructors
        Tensor();  // Default constructor

        // Constructor from vector and shape
        Tensor(const std::vector<T>& data, const std::vector<size_t>& shape);

        // Constructor for scalar tensors
        explicit Tensor(const T& value);

        // Destructor
        ~Tensor();

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
        Tensor<T> apply_elementwise_function(std::function<T(T)> func) const;

        // Matrix multiplication
        Tensor<T> matmul(const Tensor<T>& other) const;

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
    private:
        std::vector<T> data_;          // Changed from raw pointer to std::vector
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;
        size_t total_size_;

        // Helper methods
        void compute_strides();

        [[nodiscard]] std::vector<size_t> compute_broadcast_shape(
                const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) const;
        [[nodiscard]] size_t compute_flat_index(
                const std::vector<size_t>& indices, const std::vector<size_t>& strides) const;
        [[nodiscard]] std::vector<size_t> compute_indices(
                size_t flat_index, const std::vector<size_t>& shape) const;

        // Helper function to print tensor contents
        void print_tensor(std::ostream& os, size_t index, size_t depth) const;

        void broadcast_shapes(const Tensor<T>& other,
                              std::vector<size_t>& out_shape,
                              std::vector<size_t>& adjusted_strides1,
                              std::vector<size_t>& adjusted_strides2) const;

    };

    // Implementation of methods (included in header due to templates)

    // Element access (multi-dimensional indexing)
    template<typename T>
    T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
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

        auto backend = BackendManager<R>::get_backend();

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


    // Implement other methods...

} // namespace dio

#endif // GEODIO_TENSOR_H
