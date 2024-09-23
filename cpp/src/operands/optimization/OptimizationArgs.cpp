//
// Created by zwartengaten on 9/22/24.
//

#include "OptimizationArgs.h"

namespace dio {
    bool OptimizationArgs::has(const std::string &key) const {
        return args_.find(key) != args_.end();
    }

    void OptimizationArgs::set(const std::string &key, OptimizationArgs::ArgType value) {
        args_[key] = value;
    }
} // dio