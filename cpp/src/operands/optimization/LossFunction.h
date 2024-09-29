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
#ifndef GEODIO_LOSSFUNCTION_H
#define GEODIO_LOSSFUNCTION_H

#include <cmath>
#include "../../tensors/AnyTensor.h"
#include "OptimizationArgs.h"

namespace dio {

class LossFunctions {
public:
    static a_tens mean_squared_error(const a_tens& prediction, const a_tens& target);

    static a_tens cross_entropy(const a_tens& prediction, const a_tens& target);

    static a_tens compute_loss(const a_tens& prediction, const a_tens& target, LossFunction loss_func);

    static a_tens mean_squared_error_gradient(const a_tens& prediction, const a_tens& target);

    static a_tens cross_entropy_gradient(const a_tens& prediction, const a_tens& target);

    static a_tens compute_loss_gradient(const a_tens& prediction, const a_tens& target, LossFunction loss_func);
};

} // namespace dio

#endif //GEODIO_LOSSFUNCTION_H
