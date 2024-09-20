// operations.cpp

#include "OperationRegistry.h"
#include <cmath>



namespace dio {

tensor_ptr add_forward (const std::vector<tensor_ptr>& inputs) {
    auto tensor_result = (inputs[0]->add(*inputs[1]));
    auto result = std::make_shared<AnyTensor>(tensor_result);
    return result;
}

std::vector<tensor_ptr> add_backward(
        const std::vector<tensor_ptr>& inputs,
        const tensor_ptr& upstream_gradient,
        const tensor_ptr& /*forward_output*/) {
   return {upstream_gradient, upstream_gradient};
}

tensor_ptr multiply_forward (const std::vector<tensor_ptr>& inputs) {
    auto tensor_result = (inputs[0]->multiply(*inputs[1]));
    auto result = std::make_shared<AnyTensor>(tensor_result);
    return result;
}

std::vector<tensor_ptr> multiply_backward(
        const std::vector<tensor_ptr>& inputs,
        const tensor_ptr& upstream_gradient,
        const tensor_ptr& /*forward_output*/) {
    auto result =
            std::vector<tensor_ptr>({
                   std::make_shared<AnyTensor>(upstream_gradient->multiply(*inputs[1])),
                   std::make_shared<AnyTensor>(upstream_gradient->multiply(*inputs[0]))
           });
    return result;
}

tensor_ptr sigmoid_forward (const std::vector<tensor_ptr>& inputs) {
    auto tensor_result = inputs[0]->apply_unary([](auto x) {
                return 1.0f / (1.0f + std::exp(-x));
            });
    auto result = std::make_shared<AnyTensor>(tensor_result);
    return result;
}

std::vector<tensor_ptr> sigmoid_backward(
        const std::vector<tensor_ptr>& inputs,
        const tensor_ptr& upstream_gradient,
        const tensor_ptr& forward_output) {
    auto one_minus_output = forward_output->apply_unary([](auto x) {
        return 1.0f - x;
    });
    auto local_gradient = forward_output->multiply(one_minus_output);
    auto result =  std::vector<tensor_ptr>({
       std::make_shared<AnyTensor>(upstream_gradient->multiply(local_gradient))
   });
    return result;
}


tensor_ptr lt_forward (const std::vector<tensor_ptr>& inputs) {
    auto tensor_result = (inputs[0]->multiply(*inputs[1]));
    auto result = std::make_shared<AnyTensor>(tensor_result);
    return result;
}

std::vector<tensor_ptr> lt_backward(
        const std::vector<tensor_ptr>& inputs,
        const tensor_ptr& upstream_gradient,
        const tensor_ptr& forward_output) {
   return {upstream_gradient, upstream_gradient};
}


tensor_ptr linear_forward(
    const std::vector<tensor_ptr>& inputs) {
    const auto& input = inputs[0];
    const auto& weights = inputs[1];
    const auto& bias = inputs[2];

    auto weighted_input = input->matmul(*weights); // Matrix multiplication
    auto output = weighted_input.add(*bias);      // Add bias

    return std::make_shared<AnyTensor>(output);
}

std::vector<tensor_ptr> linear_backward(
    const std::vector<tensor_ptr>& inputs,
    const tensor_ptr& upstream_gradient,
    const tensor_ptr& /*forward_output*/) {
    const auto& input = inputs[0];
    const auto& weights = inputs[1];

    // Bias is inputs[2], not needed directly for gradients
    // Gradient w.r.t Input
    auto grad_input = std::make_shared<AnyTensor>(upstream_gradient->matmul(weights->transpose()));

    // Gradient w.r.t Weights
    auto grad_weights = std::make_shared<AnyTensor>(input->transpose().matmul(*upstream_gradient));

    // Gradient w.r.t Bias
    auto grad_bias = std::make_shared<AnyTensor>(upstream_gradient->sum(/*axis=*/{0})); // Sum over batch dimension
    return {grad_input, grad_weights, grad_bias};
}


void initialize_operations() {
    OperationRegistry &registry = OperationRegistry::get_instance();

    // Register Add operation
    registry.register_operation(OperandType::Add, Operation{
            add_forward, add_backward
    });

    // Register Multiply operation
    registry.register_operation(OperandType::Multiply, Operation{
            multiply_forward, multiply_backward
    });

    // Register Sigmoid operation
    registry.register_operation(OperandType::Sigmoid, Operation{
            sigmoid_forward, sigmoid_backward
    });

    // Register Linear Transformation Operand
    registry.register_operation(OperandType::LinearTransformation, Operation{
            linear_forward, linear_backward
    });
    // Register other operations as needed
}
}

