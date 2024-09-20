#ifndef GEODIO_OPERATIONS_H
#define GEODIO_OPERATIONS_H

namespace dio {

tensor_ptr add_forward (const std::vector<tensor_ptr>& inputs);

std::vector<tensor_ptr> add_backward(
        const std::vector<tensor_ptr>& inputs,
        const tensor_ptr& upstream_gradient,
        const tensor_ptr& /*forward_output*/);

tensor_ptr multiply_forward (const std::vector<tensor_ptr>& inputs);

std::vector<tensor_ptr> multiply_backward(
        const std::vector<tensor_ptr>& inputs,
        const tensor_ptr& upstream_gradient,
        const tensor_ptr& /*forward_output*/);

tensor_ptr sigmoid_forward (const std::vector<tensor_ptr>& inputs);

std::vector<tensor_ptr> sigmoid_backward(
        const std::vector<tensor_ptr>& inputs,
        const tensor_ptr& upstream_gradient,
        const tensor_ptr& /*forward_output*/);

tensor_ptr linear_forward (const std::vector<tensor_ptr>& inputs);

std::vector<tensor_ptr> linear_backward(
    const std::vector<tensor_ptr>& inputs,
    const tensor_ptr& upstream_gradient,
    const tensor_ptr& /*forward_output*/);


void initialize_operations();
} // namespace dio

#endif //GEODIO_OPERATIONS_H
