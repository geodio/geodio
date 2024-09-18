// operations.cpp

#include "OperationRegistry.h"
#include <cmath>



namespace dio {
template<typename T>
std::shared_ptr<Tensor<T>> add_forward (const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    auto tensor_result = (inputs[0]->add(*inputs[1]));
    auto result = std::make_shared<Tensor<T>>(tensor_result);
    return result;
}

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> add_backward(
        const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
        const std::shared_ptr<Tensor<T>>& upstream_gradient,
        const std::shared_ptr<Tensor<T>>& /*forward_output*/) {
   return {upstream_gradient, upstream_gradient};
}

template<typename T>
std::shared_ptr<Tensor<T>> multiply_forward (const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    auto tensor_result = (inputs[0]->multiply(*inputs[1]));
    auto result = std::make_shared<Tensor<T>>(tensor_result);
    return result;
}

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> multiply_backward(
        const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
        const std::shared_ptr<Tensor<T>>& upstream_gradient,
        const std::shared_ptr<Tensor<T>>& /*forward_output*/) {
    auto result =
            std::vector<std::shared_ptr<Tensor<T>>>({
                   std::make_shared<Tensor<T>>(upstream_gradient->multiply(*inputs[1])),
                   std::make_shared<Tensor<T>>(upstream_gradient->multiply(*inputs[0]))
           });
    return result;
}

template<typename T>
std::shared_ptr<Tensor<T>> sigmoid_forward (const std::vector<std::shared_ptr<Tensor<T>>>& inputs) {
    auto tensor_result = inputs[0]->apply_elementwise_function([](T x) {
                return 1.0f / (1.0f + std::exp(-x));
            });
    auto result = std::make_shared<Tensor<T>>(tensor_result);
    return result;
}

template<typename T>
std::vector<std::shared_ptr<Tensor<T>>> sigmoid_backward(
        const std::vector<std::shared_ptr<Tensor<T>>>& inputs,
        const std::shared_ptr<Tensor<T>>& upstream_gradient,
        const std::shared_ptr<Tensor<T>>& forward_output) {
    auto one_minus_output = forward_output->apply_elementwise_function([](T x) {
        return 1.0f - x;
    });
    auto local_gradient = forward_output->multiply(one_minus_output);
    auto result =  std::vector<std::shared_ptr<Tensor<T>>>({
       std::make_shared<Tensor<T>>(upstream_gradient->multiply(local_gradient))
   });
    return result;
}

template<typename T>
void initialize_operations() {
    OperationRegistry<T> &registry = OperationRegistry<T>::get_instance();

    // Register Add operation
    registry.register_operation(OperandType::Add, Operation<T>{
            add_forward<T>, add_backward<T>
    });

    // Register Multiply operation
    registry.register_operation(OperandType::Multiply, Operation<T>{
            multiply_forward<T>, multiply_backward<T>
    });

    // Register Sigmoid operation
    registry.register_operation(OperandType::Sigmoid, Operation<T>{
            sigmoid_forward<T>, sigmoid_backward<T>
    });

    // Register other operations as needed
}
}
template void dio::initialize_operations<float>();
template void dio::initialize_operations<double>();
template void dio::initialize_operations<int>();
