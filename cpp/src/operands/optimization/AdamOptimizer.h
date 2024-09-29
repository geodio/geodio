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
#ifndef GEODIO_ADAMOPTIMIZER_H
#define GEODIO_ADAMOPTIMIZER_H

#include <unordered_map>
#include <cmath>
#include "../../tensors/AnyTensor.h"

namespace dio {

class AdamOptimizer {
public:
    explicit AdamOptimizer(float lr = 0.001, float beta1 = 0.9, float beta2 = 0.999, float epsilon = 1e-8)
        : learning_rate_(lr), beta1_(beta1), beta2_(beta2), epsilon_(epsilon) {}

    a_tens update(const a_tens& weights, const a_tens& gradient, int step);

private:
    float learning_rate_;
    float beta1_, beta2_, epsilon_;
    std::unordered_map<const a_tens*, a_tens> m_;  // First moment vector
    std::unordered_map<const a_tens*, a_tens> v_;  // Second moment vector
};

} // namespace dio


#endif //GEODIO_ADAMOPTIMIZER_H
