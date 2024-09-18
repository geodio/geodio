#ifndef GEODIO_OPERATIONS_H
#define GEODIO_OPERATIONS_H

namespace dio {
template<typename T>
std::shared_ptr<Tensor<T>> add_forward (const std::vector<std::shared_ptr<Tensor<T>>>& inputs);

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> add_backward(
        const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
        const std::shared_ptr<Tensor<T>>& upstream_gradient);

template<typename T>
std::shared_ptr<Tensor<T>> multiply_forward (const std::vector<std::shared_ptr<Tensor<T>>>& inputs);

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> multiply_backward(
        const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
        const std::shared_ptr<Tensor<T>>& upstream_gradient);

template<typename T>
std::shared_ptr<Tensor<T>> sigmoid_forward (const std::vector<std::shared_ptr<Tensor<T>>>& inputs);

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> sigmoid_backward(
        const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
        const std::shared_ptr<Tensor<T>>& upstream_gradient);

template<typename T>
void initialize_operations();
} // namespace dio

#endif //GEODIO_OPERATIONS_H
