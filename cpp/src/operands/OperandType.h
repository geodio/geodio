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
#ifndef GEODIO_OPERANDTYPE_H
#define GEODIO_OPERANDTYPE_H

#include <string>

namespace dio{
    enum class OperandType {
        Constant,
        Variable,
        Weight,
        Seq,        // Sequential execution
        Tick,       // Increment time
        getTime,    // Return current time wrapped in tensor
        Jump,       // To jump there!
        Label,      // Where to jump!
        Condition,  // General conditional operand
        LessThan,   // <
        GreaterThan,  // >
        Equal,      // ==
        LessThanOrEqual, // <=
        GreaterThanOrEqual, // >=
        Add,
        Multiply,
        Sigmoid,
        LinearTransformation,
        Identity,
        RETURN, //stops the execution of the CG
        SET, // state = 4, or state[4] = 10
        ACCESS, // weight[0], or weight[1:2, 3:]
        Function,
        FunctionCall,
        FunctionPtr,
        String
        // Add more operations as needed
    };

    std::string operand_type_to_string(OperandType type);




} // namespace dio

#endif //GEODIO_OPERANDTYPE_H
