#include "ComputationalGraph.h"
#ifndef GEODIO_EXECUTIONENGINE_H
#define GEODIO_EXECUTIONENGINE_H

namespace dio {
    class ExecutionEngine {
    public:
        static std::shared_ptr<Tensor<float>> forward(ComputationalGraph& graph, int output_operand_id);
        static void backward(ComputationalGraph& graph, int output_operand_id);

    private:
        static std::unordered_map<int, std::shared_ptr<Tensor<float>>> forward_cache;
        static void compute_forward(ComputationalGraph& graph, int operand_id);
        static void compute_backward(ComputationalGraph& graph, int operand_id, const std::shared_ptr<Tensor<float>>& upstream_gradient);
    };

} // namespace dio

#endif //GEODIO_EXECUTIONENGINE_H
