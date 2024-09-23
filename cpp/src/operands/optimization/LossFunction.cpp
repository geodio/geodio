//
// Created by zwartengaten on 9/22/24.
//

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