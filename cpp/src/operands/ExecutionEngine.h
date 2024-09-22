#include "ComputationalGraph.h"
#include "AnyTensor.h"
#include "optimization/OptimizationArgs.h"

#ifndef GEODIO_EXECUTIONENGINE_H
#define GEODIO_EXECUTIONENGINE_H

namespace dio {
class ExecutionEngine {
public:
    static a_tens forward(ComputationalGraph& graph, int output_operand_id, const std::vector<a_tens>& args = {});
    static void backward(ComputationalGraph& graph,
                         int output_operand_id,
                         const a_tens& loss_gradient,
                         const std::vector<a_tens>& args = {});

    static void optimize(ComputationalGraph &graph, const a_tens &input_data, const a_tens &target_data,
             OptimizationArgs &args);

private:
    static void compute_forward(ComputationalGraph& graph, int operand_id,
                                std::unordered_map<int, a_tens>& forward_cache, const std::vector<a_tens>& args = {});
    static void compute_backward(ComputationalGraph& graph, int operand_id,
                                 const a_tens& upstream_gradient,
                                 std::unordered_map<int, a_tens>& forward_cache,
                                 std::unordered_map<int, a_tens>& gradient_cache);

};
} // namespace dio

#endif //GEODIO_EXECUTIONENGINE_H
