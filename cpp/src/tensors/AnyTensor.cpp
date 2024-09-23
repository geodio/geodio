
#include "AnyTensor.h"

namespace dio {
[[nodiscard]] AnyTensor AnyTensor::transpose(const std::vector<size_t>& axis) const {
    // Dispatch based on the tensor's actual type
    if (is<float>()) {
        return AnyTensor(std::make_shared<Tensor<float>>(get<float>().transpose(axis)));
    } else if (is<double>()) {
        return AnyTensor(std::make_shared<Tensor<double>>(get<double>().transpose(axis)));
    } else if (is<int>()) {
        return AnyTensor(std::make_shared<Tensor<int>>(get<int>().transpose(axis)));
    } else {
        throw std::runtime_error("Unsupported tensor type for transpose");
    }
}

[[nodiscard]] AnyTensor AnyTensor::sum(const std::vector<size_t>& axis) const {
    // Dispatch based on the tensor's actual type
    if (is<float>()) {
        return AnyTensor(std::make_shared<Tensor<float>>(get<float>().sum(axis)));
    } else if (is<double>()) {
        return AnyTensor(std::make_shared<Tensor<double>>(get<double>().sum(axis)));
    } else if (is<int>()) {
        return AnyTensor(std::make_shared<Tensor<int>>(get<int>().sum(axis)));
    } else {
        throw std::runtime_error("Unsupported tensor type for sum");
    }
}

[[nodiscard]] AnyTensor AnyTensor::slice(const std::vector<Slice>& slices) const {
    if (is<float>()) {
        return AnyTensor(std::make_shared<Tensor<float>>(get<float>().slice(slices)));
    } else if (is<double>()) {
        return AnyTensor(std::make_shared<Tensor<double>>(get<double>().slice(slices)));
    } else if (is<int>()) {
        return AnyTensor(std::make_shared<Tensor<int>>(get<int>().slice(slices)));
    } else {
        throw std::runtime_error("Unsupported tensor type for slicing");
    }
}

const std::vector<size_t> &AnyTensor::shape() const {
        // Dispatch based on the tensor's actual type
    if (is<float>()) {
        return get<float>().shape();
    } else if (is<double>()) {
        return get<double>().shape();
    } else if (is<int>()) {
        return get<int>().shape();
    } else {
        throw std::runtime_error("Unsupported tensor type for shape");
    }
}

// Operators
a_tens a_tens::operator+(const a_tens& other) const {
    return this->add(other);
}

a_tens a_tens::operator-(const a_tens& other) const {
    return this->subtract(other);
}

a_tens a_tens::operator*(const a_tens& other) const {
    return this->multiply(other);
}

a_tens a_tens::operator/(const a_tens& other) const {
    return this->divide(other);
}


} // dio