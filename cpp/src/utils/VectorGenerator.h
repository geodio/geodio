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
#ifndef GEODIO_VECTORGENERATOR_H
#define GEODIO_VECTORGENERATOR_H

#include <vector>
#include <random>
#include <cmath>
#include <iostream>
#include <limits>

namespace dio {
    // Class for generating vectors with arbitrary values and distributions
    class VectorGenerator {
    public:
        // Generate a vector of length n initialized with zeros
        template <typename T>
        std::vector<T> zeros(size_t n) {
            return std::vector<T>(n, static_cast<T>(0));
        }

        // Generate a vector of length n initialized with ones
        template <typename T>
        std::vector<T> ones(size_t n) {
            return std::vector<T>(n, static_cast<T>(1));
        }

        // Generate a vector of length n with uniform distribution in range [a, b]
        template <typename T>
        std::vector<T> uniform(size_t n, T a, T b) {
            std::vector<T> vec(n);
            std::mt19937 gen(rd());
            std::uniform_real_distribution<T> dis(a, b);

            for (size_t i = 0; i < n; ++i) {
                vec[i] = dis(gen);
            }
            return vec;
        }

        // Generate a vector of length n with normal distribution in range [a, b]
        template <typename T>
        std::vector<T> normal(size_t n, T a, T b) {
            std::vector<T> vec(n);
            std::mt19937 gen(rd());

            // Ensure that the range [a, b] is correctly handled. Use the midpoint and a reasonable standard deviation.
            T mean = (a + b) / 2;
            T stddev = (b - a) / 6;  // A typical normal distribution has ~99.7% of values within ±3 standard deviations

            if (stddev <= std::numeric_limits<T>::epsilon()) {
                stddev = static_cast<T>(1);  // Avoid division by zero or very small stddev
            }

            std::normal_distribution<T> dis(mean, stddev);

            for (size_t i = 0; i < n; ++i) {
                vec[i] = dis(gen);
                // Check for NaN and replace with mean if needed
                if (std::isnan(vec[i])) {
                    vec[i] = mean;
                }
                // Clamp values to stay within [a, b]
                if (vec[i] < a) vec[i] = a;
                if (vec[i] > b) vec[i] = b;
            }
            return vec;
        }

        // Generate a vector of length n with uniform distribution in range [-1, 1]
        template <typename T>
        std::vector<T> uniform_11(size_t n) {
            return uniform(n, static_cast<T>(-1), static_cast<T>(1));
        }

        // Generate a vector of length n with normal distribution in range [-1, 1]
        template <typename T>
        std::vector<T> normal_11(size_t n) {
            return normal(n, static_cast<T>(-1), static_cast<T>(1));
        }

    private:
        std::random_device rd;
    };

}

#endif //GEODIO_VECTORGENERATOR_H
