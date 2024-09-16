#ifndef GEOPY_TENSOR_H
#define GEOPY_TENSOR_H

#include <initializer_list>
#include <vector>
#include <cstddef>
#include <string>
#include <sstream>

namespace dio {

    template<typename T>
    class Tensor {
    public:
        // Constructors
        Tensor(std::initializer_list<T> list);
        Tensor(const std::vector<T>& data, const std::vector<size_t>& shape);
        Tensor(const T* data, const std::vector<size_t>& shape);

        // Destructor
        ~Tensor() = default;

        // Information methods
        size_t num_dimensions() const;
        const std::vector<size_t>& shape() const;

        // Broadcasting
        void broadcast(const std::vector<size_t>& new_shape);

        // Element access
        T& operator()(const std::vector<size_t>& indices);
        const T& operator()(const std::vector<size_t>& indices) const;

        // File I/O
        void save_to_file(const std::string& filename) const;
        void load_from_file(const std::string& filename);

        // To string representation
        // Method to get a string representation of the tensor
        std::string to_string() const;


        // Element-wise operations
        Tensor<T> operator+(const Tensor<T>& other) const;
        Tensor<T> operator*(const Tensor<T>& other) const;

        // GPU Support (later, for example with CUDA)
        virtual void to_gpu();    // Transfer data to GPU
        virtual void from_gpu();  // Transfer data back to CPU

        // Friend operator<< for printing
        template<typename U>
        friend std::ostream& operator<<(std::ostream& os, const Tensor<U>& tensor);


    private:
        std::vector<size_t> shape_;
        std::vector<size_t> strides_;

        // Helper methods
        void compute_strides();
        void initialize_from_list(const std::initializer_list<T>& list);

    protected:
        size_t compute_size(const std::vector<size_t>& shape) const;

        std::vector<T> data_;
    };

    template<typename T>
    std::string Tensor<T>::to_string() const {
        std::ostringstream ss;
        ss << *this;  // Utilize operator<< to get the string
        return ss.str();
    }

    // Implementation of the << operator to print tensor contents
    template<typename T>
    std::ostream& operator<<(std::ostream& os, const Tensor<T>& tensor) {
        os << "Tensor(";
        if (tensor.data_.empty()) {
            os << "empty";
        } else {
            size_t total_elements = tensor.compute_size(tensor.shape_);
            os << "[";
            for (size_t i = 0; i < total_elements; ++i) {
                os << tensor.data_[i];
                if (i < total_elements - 1) os << ", ";
            }
            os << "]";
        }
        os << "), shape=";
        os << "[";
        for (size_t i = 0; i < tensor.shape_.size(); ++i) {
            os << tensor.shape_[i];
            if (i < tensor.shape_.size() - 1) os << ", ";
        }
        os << "]";
        return os;
    }


} // dio

#endif //GEOPY_TENSOR_H
