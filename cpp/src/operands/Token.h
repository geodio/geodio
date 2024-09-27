#ifndef GEODIO_TOKEN_H
#define GEODIO_TOKEN_H

namespace dio {
    enum class TokenType {
        Operand,        // Represents an operand to be processed
        Return,         // Represents a return instruction for where to store the result
        Jump,           // Represents a jump instruction, with condition
        Label,          // Represents a conditional jump
        ConditionReturn,
        ConditionResult
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
