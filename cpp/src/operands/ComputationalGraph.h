#include "OperandType.h"
#include "Operand.h"
#include "AnyTensor.h"

#ifndef GEODIO_COMPUTATIONALGRAPH_H
#define GEODIO_COMPUTATIONALGRAPH_H

namespace dio{
class ComputationalGraph {
public:
    std::unordered_map<int, Operand> operands{};
    std::unordered_map<int, tensor_ptr> weights{}; // Weights associated with operands
    std::unordered_map<int, tensor_ptr> gradients{}; // Gradients computed during backpropagation

    // Methods to add operands, weights, etc.
};
} // namespace dio

#endif //GEODIO_COMPUTATIONALGRAPH_H
