#include "OperandType.h"
#include "Operand.h"
#include "../tensors/AnyTensor.h"

#ifndef GEODIO_COMPUTATIONALGRAPH_H
#define GEODIO_COMPUTATIONALGRAPH_H

namespace dio{
class ComputationalGraph {
public:
    explicit ComputationalGraph(int rootId=0) : root_id(rootId) {}

    std::unordered_map<int, Operand> operands{};
    std::unordered_map<int, a_tens> weights{}; // Weights associated with operands
    std::unordered_map<int, a_tens> gradients{}; // Gradients computed during backpropagation

    std::unordered_map<int, a_tens> constants{}; // Constants associated with operands
    std::unordered_map<int, int> var_map{}; // Mapping from Variable_id to index of provided arguments
    int root_id;
    // Methods to add operands, weights, etc.
};
} // namespace dio

#endif //GEODIO_COMPUTATIONALGRAPH_H
