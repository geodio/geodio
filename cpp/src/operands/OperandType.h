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
        LinearTransformation
        // Add more operations as needed
    };

    std::string operand_type_to_string(OperandType type);




} // namespace dio

#endif //GEODIO_OPERANDTYPE_H
