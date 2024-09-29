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
#ifndef GEODIO_OPTIMIZATIONARGS_H
#define GEODIO_OPTIMIZATIONARGS_H
#include <unordered_map>
#include <string>
#include <variant>
#include <iostream>

namespace dio {

enum class LossFunction {
    MeanSquaredError,
    CrossEntropy,
    Custom
};

enum class GradientRegularizer {
    Adam,
    SGD,
    Custom
};

class OptimizationArgs {
public:
    using ArgType = std::variant<int, float, double, LossFunction, GradientRegularizer>;

    void set(const std::string& key, ArgType value);

    template <typename T>
    T get(const std::string& key) const;

    bool has(const std::string& key) const;

private:
    std::unordered_map<std::string, ArgType> args_;
};

    template<typename T>
    T OptimizationArgs::get(const std::string &key) const {
        if (args_.find(key) == args_.end()) {
            throw std::invalid_argument("Key not found: " + key);
        }
        return std::get<T>(args_.at(key));
    }

} // dio

#endif //GEODIO_OPTIMIZATIONARGS_H
