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

namespace dio {
    std::string operand_type_to_string(OperandType type) {
        switch (type) {
            case OperandType::Constant:
                return "Constant";
            case OperandType::Variable:
                return "Variable";
            case OperandType::Weight:
                return "Weight";
            case OperandType::Seq:
                return "Sequential Execution";
            case OperandType::Tick:
                return "Increment Time";
            case OperandType::getTime:
                return "Get Time";
            case OperandType::Jump:
                return "Jump";
            case OperandType::Label:
                return "Label";
            case OperandType::Condition:
                return "Condition";
            case OperandType::LessThan:
                return "LessThan";
            case OperandType::GreaterThan:
                return "GreaterThan";
            case OperandType::Equal:
                return "Equal";
            case OperandType::LessThanOrEqual:
                return "LessThanOrEqual";
            case OperandType::GreaterThanOrEqual:
                return "GreaterThanOrEqual";
            case OperandType::Add:
                return "Add";
            case OperandType::Multiply:
                return "Multiply";
            case OperandType::Sigmoid:
                return "Sigmoid";
            case OperandType::LinearTransformation:
                return "Linear Transformation";
            default:
                return "Unknown OperandType";
        }
    }
} // dio