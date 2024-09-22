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
