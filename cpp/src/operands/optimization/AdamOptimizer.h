#ifndef GEODIO_ADAMOPTIMIZER_H
#define GEODIO_ADAMOPTIMIZER_H

#include <unordered_map>
#include <cmath>
#include "AnyTensor.h"

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
