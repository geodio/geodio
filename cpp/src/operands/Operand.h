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
