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

#include "LossFunction.h"

namespace dio {
    a_tens LossFunctions::compute_loss_gradient(const a_tens &prediction, const a_tens &target, LossFunction loss_func) {
        switch (loss_func) {
            case LossFunction::MeanSquaredError:
                return mean_squared_error_gradient(prediction, target);
            case LossFunction::CrossEntropy:
                return cross_entropy_gradient(prediction, target);
            default:
                throw std::invalid_argument("Unsupported loss function");
        }
    }

    a_tens LossFunctions::cross_entropy_gradient(const a_tens &prediction, const a_tens &target) {
        return target.divide(prediction) * a_tens(-1.0f);  // Gradient of cross-entropy w.r.t. prediction
    }

    a_tens LossFunctions::mean_squared_error_gradient(const a_tens &prediction, const a_tens &target) {
        return a_tens(2.0f) * (prediction - target);  // Gradient of MSE w.r.t. prediction
    }

    a_tens LossFunctions::compute_loss(const a_tens &prediction, const a_tens &target, LossFunction loss_func) {
        switch (loss_func) {
            case LossFunction::MeanSquaredError:
                return mean_squared_error(prediction, target);
            case LossFunction::CrossEntropy:
                return cross_entropy(prediction, target);
            default:
                throw std::invalid_argument("Unsupported loss function");
        }
    }

    a_tens LossFunctions::cross_entropy(const a_tens &prediction, const a_tens &target) {
        std::function<float (float)> func = [](float x) { return std::log(x); };
        a_tens log_pred = prediction.apply_unary(func);
        return target.multiply(log_pred).sum();  // Negative log likelihood
    }

    a_tens LossFunctions::mean_squared_error(const a_tens &prediction, const a_tens &target) {
        a_tens diff = prediction - target;
        return diff.multiply(diff).sum();  // Sum of squared differences
    }
} // dio