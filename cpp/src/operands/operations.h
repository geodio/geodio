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
#ifndef GEODIO_OPERATIONS_H
#define GEODIO_OPERATIONS_H

namespace dio {

class OperationInitializer {
public:
    static void initialize();
};

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
