#ifndef GEODIO_OPERANDTYPE_H
#define GEODIO_OPERANDTYPE_H

namespace dio{
    enum class OperandType {
        Constant,
        Variable,
        Weight,
        Add,
        Multiply,
        Sigmoid,
        LinearTransformation
        // Add more operations as needed
    };
} // namespace dio

#endif //GEODIO_OPERANDTYPE_H
