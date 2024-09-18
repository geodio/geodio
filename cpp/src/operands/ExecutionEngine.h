#include "ComputationalGraph.h"
#ifndef GEODIO_EXECUTIONENGINE_H
#define GEODIO_EXECUTIONENGINE_H

namespace dio {
template<typename T>
class ExecutionEngine {
public:
    static std::shared_ptr<Tensor<T>> forward(ComputationalGraph<T>& graph, int output_operand_id);
    static void backward(ComputationalGraph<T>& graph,
                         int output_operand_id,
                         const std::shared_ptr<Tensor<T>>& loss_gradient);

private:
    static void compute_forward(ComputationalGraph<T>& graph, int operand_id,
                                std::unordered_map<int, std::shared_ptr<Tensor<T>>>& forward_cache);
    static void compute_backward(ComputationalGraph<T>& graph, int operand_id,
                                 const std::shared_ptr<Tensor<T>>& upstream_gradient,
                                 std::unordered_map<int, std::shared_ptr<Tensor<T>>>& forward_cache,
                                 std::unordered_map<int, std::shared_ptr<Tensor<T>>>& gradient_cache);
};
} // namespace dio

#endif //GEODIO_EXECUTIONENGINE_H
