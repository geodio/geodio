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
#include "OperationRegistry.h"

namespace dio {
OperationRegistry& OperationRegistry::get_instance() {
    static OperationRegistry instance;
    return instance;
}

void OperationRegistry::register_operation(OperandType type, const Operation& operation) {
    operations_[type] = operation;
}

Operation OperationRegistry::get_operation(OperandType type) const {
    auto it = operations_.find(type);
    if (it != operations_.end()) {
        return it->second;
    } else {
        throw std::runtime_error(
            std::string("Operation not registered for OperandType in OperationRegistry"));
    }
}
} // namespace dio
