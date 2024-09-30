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
#ifndef GEODIO_TOKEN_H
#define GEODIO_TOKEN_H

namespace dio {
    enum class TokenType {
        Operand,        // Represents an operand to be processed
        Return,         // Represents a return instruction for where to store the result
        Jump,           // Represents a jump instruction, with condition
        Label,          // Represents a conditional jump
        ConditionReturn,
        ConditionResult,
        END,
        START
    };

    class Token {
    public:
        TokenType type;
        int operand_id;         // For Operand and Return tokens
        int return_location;    // For Return tokens
        int jump_label;         // For Jump and ConditionalJump tokens
        bool condition;         // For ConditionalJump tokens

        explicit Token(TokenType type, int operand_id = -1, int return_location = -1,
                       int jump_label = -1, bool condition = false)
            : type(type), operand_id(operand_id), return_location(return_location),
            jump_label(jump_label), condition(condition) {}
    };

} // dio

#endif //GEODIO_TOKEN_H
