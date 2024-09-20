//
// Created by zwartengaten on 9/20/24.
//

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

} // dio