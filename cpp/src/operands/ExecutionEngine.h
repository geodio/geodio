#include "ComputationalGraph.h"
#include "AnyTensor.h"

#ifndef GEODIO_EXECUTIONENGINE_H
#define GEODIO_EXECUTIONENGINE_H

namespace dio {
class ExecutionEngine {
public:
    static tensor_ptr forward(ComputationalGraph& graph, int output_operand_id);
    static void backward(ComputationalGraph& graph,
                         int output_operand_id,
                         const tensor_ptr& loss_gradient);

private:
    static void compute_forward(ComputationalGraph& graph, int operand_id,
                                std::unordered_map<int, tensor_ptr>& forward_cache);
    static void compute_backward(ComputationalGraph& graph, int operand_id,
                                 const tensor_ptr& upstream_gradient,
                                 std::unordered_map<int, tensor_ptr>& forward_cache,
                                 std::unordered_map<int, tensor_ptr>& gradient_cache);
};
} // namespace dio

#endif //GEODIO_EXECUTIONENGINE_H
