
#ifndef GEODIO_LOSSFUNCTION_H
#define GEODIO_LOSSFUNCTION_H

#include <cmath>
#include "AnyTensor.h"
#include "OptimizationArgs.h"

namespace dio {

class LossFunctions {
public:
    static a_tens mean_squared_error(const a_tens& prediction, const a_tens& target) {
        a_tens diff = prediction - target;
        return diff.multiply(diff).sum();  // Sum of squared differences
    }

    static a_tens cross_entropy(const a_tens& prediction, const a_tens& target) {
        std::function<float (float)> func = [](float x) { return std::log(x); };
        a_tens log_pred = prediction.apply_unary(func);
        return target.multiply(log_pred).sum();  // Negative log likelihood
    }

    static a_tens compute_loss(const a_tens& prediction, const a_tens& target, LossFunction loss_func) {
        switch (loss_func) {
            case LossFunction::MeanSquaredError:
                return mean_squared_error(prediction, target);
            case LossFunction::CrossEntropy:
                return cross_entropy(prediction, target);
            default:
                throw std::invalid_argument("Unsupported loss function");
        }
    }
};

} // namespace dio

#endif //GEODIO_LOSSFUNCTION_H
