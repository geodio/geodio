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
#include <memory>
#include "../tensors/Tensor.h"
#include "../tensors/AnyTensor.h"

#ifndef GEODIO_OPERATION_H
#define GEODIO_OPERATION_H

namespace dio {

using tensor = a_tens;

using ForwardFunc = std::function<tensor(
    const std::vector<tensor>& inputs)>;

using BackwardFunc = std::function<std::vector<tensor>(
    const std::vector<tensor>& inputs,
    const tensor& upstream_gradient,
    const tensor& forward_output)>;

struct Operation {
    ForwardFunc forward;
    BackwardFunc backward;
};

} // namespace dio


#endif //GEODIO_OPERATION_H
