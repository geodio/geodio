#ifndef GEODIO_TENSOR_H
#define GEODIO_TENSOR_H

#include "../backends/Backend.h"
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
        [[nodiscard]] size_t compute_size(const std::vector<size_t>& shape) const;

        std::vector<T> get_data() const;
        // Element access
        T& operator()(const std::vector<size_t>& indices);

        const T& operator()(const std::vector<size_t>& indices) const;
        // Element-wise operations
        Tensor<T> operator+(const Tensor<T>& other) const;

        Tensor<T> operator*(const Tensor<T>& other) const;

        // To string representation
        [[nodiscard]] std::string to_string() const;

        // Friend operator<< for printing
        template<typename U>
        friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);

        // GPU operations
        void to_device();
        void from_device();
    private:
        T* data_;
        std::vector<size_t> shape_;

        std::vector<size_t> strides_;

        size_t total_size_;

        std::shared_ptr<Backend<T>> backend_;
        // Helper methods
        void compute_strides();

        [[nodiscard]] std::vector<size_t> compute_broadcast_shape(const std::vector<size_t>& shape1, const std::vector<size_t>& shape2) const;
        [[nodiscard]] size_t compute_flat_index(const std::vector<size_t>& indices, const std::vector<size_t>& strides) const;
        [[nodiscard]] std::vector<size_t> compute_indices(size_t flat_index, const std::vector<size_t>& shape) const;

        // Helper function to print tensor contents
        void print_tensor(std::ostream& os, size_t index, size_t depth) const;

    };

    // Implementation of methods (included in header due to templates)

    // Compute strides based on shape
    template<typename T>
    void Tensor<T>::compute_strides() {
        strides_.resize(shape_.size());
        size_t stride = 1;
        for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
            strides_[i] = stride;
            stride *= shape_[i];
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

    // Element access (multi-dimensional indexing)
    template<typename T>
    T& Tensor<T>::operator()(const std::vector<size_t>& indices) {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices must match number of dimensions.");
        }
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
            index += indices[i] * strides_[i];
        }
        return data_[index];
    }

    template<typename T>
    const T& Tensor<T>::operator()(const std::vector<size_t>& indices) const {
        if (indices.size() != shape_.size()) {
            throw std::invalid_argument("Number of indices must match number of dimensions.");
        }
        size_t index = 0;
        for (size_t i = 0; i < indices.size(); ++i) {
            if (indices[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds.");
            }
            index += indices[i] * strides_[i];
        }
        return data_[index];
    }


    // Compute total size from shape
    template<typename T>
    size_t Tensor<T>::compute_size(const std::vector<size_t>& shape) const {
        return std::accumulate(shape.begin(), shape.end(), static_cast<size_t>(1), std::multiplies<>());
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


} // namespace dio

#endif // GEODIO_TENSOR_H
