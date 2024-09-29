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
#include "AdamOptimizer.h"
#include <algorithm> // for std::max

namespace dio {

    a_tens AdamOptimizer::update(const a_tens& weights, const a_tens& gradient, int step) {
        if (m_.find(&weights) == m_.end()) {
            m_[&weights] = weights.subtract(weights);  // Initialize m
            v_[&weights] = weights.subtract(weights);  // Initialize v
        }

        // Update biased first moment estimate
        m_[&weights] = m_[&weights].multiply(beta1_) + gradient.multiply(1.0f - beta1_);

        // Update biased second moment estimate
        v_[&weights] = v_[&weights].multiply(beta2_) + gradient.multiply(gradient).multiply(1.0f - beta2_);

        // Correct bias for first and second moments
        float beta1_pow = 1.0f - static_cast<float>(std::pow(beta1_, step));  // Prevent division by small values
        float beta2_pow = 1.0f - static_cast<float>(std::pow(beta2_, step));  // Prevent division by small values

        a_tens m_hat = m_[&weights].divide(AnyTensor(std::make_shared<Tensor<float>>(Tensor<float>(beta1_pow))));
        a_tens v_hat = v_[&weights].divide(AnyTensor(std::make_shared<Tensor<float>>(Tensor<float>(beta2_pow))));

        // Apply sqrt to v_hat and add epsilon_ for stability
        std::function<float(float)> func = [this](float x) {
            return std::sqrt(x) + epsilon_;  // Ensure sqrt(x) is not operating on very small values
        };

        // Update weights
        auto r = weights - m_hat.multiply(learning_rate_).divide(v_hat.apply_unary(func));
        learning_rate_ = learning_rate_ * (1.0f - 1e-8f);
        return r;
    }
} // dio
