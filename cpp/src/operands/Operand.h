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
#ifndef GEODIO_OPERAND_H
#define GEODIO_OPERAND_H

#include <vector>
#include <memory>
#include <string>
#include <unordered_map>
#include <any>
#include <limits>
#include <algorithm>
#include "../tensors/Tensor.h"
#include "OperandType.h"

namespace dio {

    class Operand {
    public:
        OperandType op_type;
        int id; // Unique identifier for the operand
        std::vector<int> inputs; // IDs of input operands
        // Additional metadata if needed

        Operand(): op_type(OperandType::Constant), id(-1), inputs({}) {}

        Operand(OperandType type, int operand_id, const std::vector<int>& input_ids)
            : op_type(type), id(operand_id), inputs(input_ids) {}
    };


} // namespace dio

#endif // GEODIO_OPERAND_H
