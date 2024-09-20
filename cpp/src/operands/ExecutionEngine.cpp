#include "ExecutionEngine.h"
#include "OperationRegistry.h"
#include "Operand.h"

namespace dio {

tensor_ptr ExecutionEngine::forward(ComputationalGraph& graph, int output_operand_id) {
    std::unordered_map<int, tensor_ptr> forward_cache;
    compute_forward(graph, output_operand_id, forward_cache);
    return forward_cache[output_operand_id];
}

void ExecutionEngine::compute_forward(ComputationalGraph& graph, int operand_id,
                                      std::unordered_map<int, tensor_ptr>& forward_cache) {
    if (forward_cache.find(operand_id) != forward_cache.end()) {
        return;
    }

    Operand& operand = graph.operands[operand_id];

    if (operand.op_type == OperandType::Constant || operand.op_type == OperandType::Variable) {
        if (graph.weights.find(operand_id) != graph.weights.end()) {
            forward_cache[operand_id] = graph.weights[operand_id];
        } else {
            throw std::runtime_error("Value for operand ID " + std::to_string(operand_id) + " not found.");
        }
    } else {
        std::vector<tensor_ptr> input_tensors;
        for (int input_id : operand.inputs) {
            compute_forward(graph, input_id, forward_cache);
            input_tensors.push_back(forward_cache[input_id]);
        }

        OperationRegistry& registry = OperationRegistry::get_instance();
        Operation operation = registry.get_operation(operand.op_type);

        std::shared_ptr<AnyTensor> result = operation.forward(input_tensors);

        forward_cache[operand_id] = result;
    }
}

void ExecutionEngine::backward(ComputationalGraph& graph, int output_operand_id,
                                  const tensor_ptr& loss_gradient) {
    std::unordered_map<int, tensor_ptr> gradient_cache;
    std::unordered_map<int, tensor_ptr> forward_cache;

    // Ensure forward pass is computed
    compute_forward(graph, output_operand_id, forward_cache);

    // Start backward pass
    compute_backward(graph, output_operand_id, loss_gradient, forward_cache, gradient_cache);

    // Store gradients in the graph's gradients map
    for (const auto& pair : gradient_cache) {
        graph.gradients[pair.first] = pair.second;
    }
}

void ExecutionEngine::compute_backward(ComputationalGraph& graph, int operand_id,
                                       const tensor_ptr& upstream_gradient,
                                       std::unordered_map<int, tensor_ptr>& forward_cache,
                                       std::unordered_map<int, tensor_ptr>& gradient_cache) {
    if (gradient_cache.find(operand_id) != gradient_cache.end()) {
        // Accumulate gradient
        gradient_cache[operand_id] = std::make_shared<AnyTensor>(gradient_cache[operand_id]->add(*upstream_gradient));
        return;
    } else {
        gradient_cache[operand_id] = upstream_gradient;
    }

    Operand& operand = graph.operands[operand_id];

    if (operand.op_type == OperandType::Constant) {
        // Constants have zero gradient
        return;
    } else if (operand.op_type == OperandType::Variable) {
        // Variables are leaves; gradient is stored
        return;
    } else {
        OperationRegistry& registry = OperationRegistry::get_instance();
        Operation operation = registry.get_operation(operand.op_type);

        // Get inputs
        std::vector<tensor_ptr> input_tensors;
        for (int input_id : operand.inputs) {
            input_tensors.push_back(forward_cache[input_id]);
        }

        // Get the forward output from the cache
        auto forward_output = forward_cache[operand_id];

        // Compute local gradients
        std::vector<tensor_ptr> local_gradients =
                operation.backward(input_tensors, upstream_gradient, forward_output);

        // Recursively compute backward for inputs
        for (size_t i = 0; i < operand.inputs.size(); ++i) {
            compute_backward(graph,
                             operand.inputs[i],
                             local_gradients[i],
                             forward_cache,
                             gradient_cache);
        }
    }
}

} // namespace dio