//    Geodio Neural Network Framework - A lightweight neural network library focused on custom architecture design and optimization.
//    Copyright (C) Copyright (C) 2024 Geodio (created by Rareș Polenciuc)
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.

#include "OperationRegistry.h"
#include <cmath>



namespace dio {

a_tens add_forward (const std::vector<a_tens>& inputs) {
    auto tensor_result = (inputs[0].add(inputs[1]));
    auto result = AnyTensor(tensor_result);
    return result;
}

std::vector<a_tens> add_backward(
        const std::vector<a_tens>& inputs,
        const a_tens& upstream_gradient,
        const a_tens& /*forward_output*/) {
   return {upstream_gradient, upstream_gradient};
}

a_tens multiply_forward (const std::vector<a_tens>& inputs) {
    auto tensor_result = (inputs[0].multiply(inputs[1]));
    auto result = AnyTensor(tensor_result);
    return result;
}

std::vector<a_tens> multiply_backward(
        const std::vector<a_tens>& inputs,
        const a_tens& upstream_gradient,
        const a_tens& /*forward_output*/) {
    auto result =
            std::vector<a_tens>({
                   AnyTensor(upstream_gradient.multiply(inputs[1])),
                   AnyTensor(upstream_gradient.multiply(inputs[0]))
           });
    return result;
}

a_tens sigmoid_forward (const std::vector<a_tens>& inputs) {
    auto tensor_result = inputs[0].apply_unary([](auto x) {
                return 1.0f / (1.0f + std::exp(-x));
            });
    auto result = AnyTensor(tensor_result);
    return result;
}

std::vector<a_tens> sigmoid_backward(
        const std::vector<a_tens>& inputs,
        const a_tens& upstream_gradient,
        const a_tens& forward_output) {
    auto one_minus_output = forward_output.apply_unary([](auto x) {
        return 1.0f - x;
    });
    auto local_gradient = forward_output.multiply(one_minus_output);
    auto result =  std::vector<a_tens>({
       AnyTensor(upstream_gradient.multiply(local_gradient))
   });
    return result;
}


a_tens linear_forward(
    const std::vector<a_tens>& inputs) {
    const auto& input = inputs[0];
    const auto& weights = inputs[1];
    const auto& bias = inputs[2];

    auto weighted_input = input.matmul(weights); // Matrix multiplication
    auto output = weighted_input.add(bias);      // Add bias

    return {output};
}

std::vector<a_tens> linear_backward(
    const std::vector<a_tens>& inputs,
    const a_tens& upstream_gradient,
    const a_tens& /*forward_output*/) {
    const auto& input = inputs[0];
    const auto& weights = inputs[1];

    // Bias is inputs[2], not needed directly for gradients
    // Gradient w.r.t Input
    auto grad_input = AnyTensor(upstream_gradient.matmul(weights.transpose()));

    // Gradient w.r.t Weights
    auto grad_weights = AnyTensor(input.transpose().matmul(upstream_gradient));

    // Gradient w.r.t Bias
    auto grad_bias = AnyTensor(upstream_gradient.sum(/*axis=*/{0})); // Sum over batch dimension
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

