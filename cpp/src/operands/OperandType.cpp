
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