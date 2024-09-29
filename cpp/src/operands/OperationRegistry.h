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
#include "OperandType.h"
#include "Operation.h"
#ifndef GEODIO_OPERATIONREGISTRY_H
#define GEODIO_OPERATIONREGISTRY_H

namespace dio {

class OperationRegistry {
public:
    static OperationRegistry& get_instance();

    void register_operation(OperandType type, const Operation& operation);
    [[nodiscard]] Operation get_operation(OperandType type) const;

private:
    OperationRegistry() = default;
    std::unordered_map<OperandType, Operation> operations_{};
};
} // namespace dio

#endif //GEODIO_OPERATIONREGISTRY_H
