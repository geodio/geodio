#ifndef GEODIO_OPERATIONS_H
#define GEODIO_OPERATIONS_H

namespace dio {

a_tens add_forward (const std::vector<a_tens>& inputs);

std::vector<a_tens> add_backward(
        const std::vector<a_tens>& inputs,
        const a_tens& upstream_gradient,
        const a_tens& /*forward_output*/);

a_tens multiply_forward (const std::vector<a_tens>& inputs);

std::vector<a_tens> multiply_backward(
        const std::vector<a_tens>& inputs,
        const a_tens& upstream_gradient,
        const a_tens& /*forward_output*/);

a_tens sigmoid_forward (const std::vector<a_tens>& inputs);

std::vector<a_tens> sigmoid_backward(
        const std::vector<a_tens>& inputs,
        const a_tens& upstream_gradient,
        const a_tens& /*forward_output*/);

a_tens linear_forward (const std::vector<a_tens>& inputs);

std::vector<a_tens> linear_backward(
    const std::vector<a_tens>& inputs,
    const a_tens& upstream_gradient,
    const a_tens& /*forward_output*/);


void initialize_operations();
} // namespace dio

#endif //GEODIO_OPERATIONS_H
