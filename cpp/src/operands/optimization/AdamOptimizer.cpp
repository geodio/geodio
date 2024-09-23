//
// Created by zwartengaten on 9/22/24.
//

#include "AdamOptimizer.h"

namespace dio {

    a_tens AdamOptimizer::update(const a_tens& weights, const a_tens& gradient, int step) {
        if (m_.find(&weights) == m_.end()) {
            m_[&weights] = a_tens(weights);  // Initialize m
            v_[&weights] = a_tens(weights);  // Initialize v
        }
        // Update biased first moment estimate
        m_[&weights] = m_[&weights].multiply(beta1_) + gradient.multiply(1 - beta1_);

        // Update biased second moment estimate
        v_[&weights] = v_[&weights].multiply(beta2_) + gradient.multiply(gradient).multiply(1 - beta2_);

        // Correct bias for first and second moments
        a_tens m_hat = m_[&weights].divide(AnyTensor(std::make_shared<Tensor<float>>(Tensor<float>(1.0f - std::pow(beta1_, step)))));
        a_tens v_hat = v_[&weights].divide(AnyTensor(std::make_shared<Tensor<float>>(Tensor<float>(1.0f - std::pow(beta2_, step)))));
        // Update weights
        std::function<float (float)> func = [this](float x) { return std::sqrt(x) + epsilon_; };
        auto r = weights - m_hat.multiply(learning_rate_).divide(v_hat.apply_unary(func));
        std::cout << "        [ORIGINAL WEIGHT: " << weights.get<float>() << std::endl;
        std::cout << "        GRADIENT: " << gradient.get<float>() << "]" << std::endl;
        return r;
    }
} // dio