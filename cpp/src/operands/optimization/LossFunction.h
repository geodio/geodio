
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
