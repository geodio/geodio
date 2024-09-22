#include "ExecutionEngine.h"
#include "OperationRegistry.h"
#include "Operand.h"
#include "optimization/OptimizationArgs.h"
#include "optimization/AdamOptimizer.h"
#include "optimization/LossFunction.h"

namespace dio {

a_tens ExecutionEngine::forward(ComputationalGraph& graph, int output_operand_id,
                                const std::vector<a_tens>& args) {
    std::unordered_map<int, a_tens> forward_cache;
    compute_forward(graph, output_operand_id, forward_cache, args);
    return forward_cache[output_operand_id];
}

void ExecutionEngine::compute_forward(ComputationalGraph& graph, int operand_id,
                                      std::unordered_map<int, a_tens>& forward_cache,
                                      const std::vector<a_tens>& args) {
    // Check if the operand's result is already cached
    if (forward_cache.find(operand_id) != forward_cache.end()) {
        return;  // Cached, no need to recompute
    }

    Operand& operand = graph.operands[operand_id];

    // Handle operand types: Constant, Weight, Variable, and Operators (Function-type)
    switch (operand.op_type) {
        case OperandType::Constant: {
            // Check if constant exists in the graph
            if (graph.constants.find(operand_id) != graph.constants.end()) {
                forward_cache[operand_id] = graph.constants[operand_id];
            } else {
                throw std::runtime_error("Constant value for operand ID "
                + std::to_string(operand_id) + " not found in the graph.");
            }
            break;
        }

        case OperandType::Weight: {
            // Check if weight exists in the graph
            if (graph.weights.find(operand_id) != graph.weights.end()) {
                forward_cache[operand_id] = graph.weights[operand_id];
            } else {
                throw std::runtime_error("Weight value for operand ID "
                + std::to_string(operand_id) + " not found in the graph.");
            }
            break;
        }

        case OperandType::Variable: {
            // Check if variable exists in the input arguments
            if (graph.var_map.find(operand_id) != graph.var_map.end()) {
                int args_index = graph.var_map[operand_id];

                // Ensure the variable index is within the bounds of the provided arguments
                if (args_index < 0 || static_cast<size_t>(args_index) >= args.size()) {
                    throw std::runtime_error("Variable for operand ID " + std::to_string(operand_id) +
                                             " (index: " + std::to_string(args_index) + ") out of bounds.");
                }

                // Cache the corresponding input argument as the forward result
                forward_cache[operand_id] = args[args_index];
            } else {
                throw std::runtime_error("Variable for operand ID "
                + std::to_string(operand_id) + " not mapped in the var_map.");
            }
            break;
        }

        default: {
            // Operator (Function type), need to compute its result by first computing its inputs
            std::vector<a_tens> input_tensors;
            for (int input_id : operand.inputs) {
                // Recursively compute forward for each input
                compute_forward(graph, input_id, forward_cache, args);
                input_tensors.push_back(forward_cache[input_id]);
            }

            // Retrieve the operation based on the operand type
            OperationRegistry& registry = OperationRegistry::get_instance();
            Operation operation = registry.get_operation(operand.op_type);

            // Compute the forward pass for this operation
            a_tens result = operation.forward(input_tensors);

            // Cache the result
            forward_cache[operand_id] = result;
            break;
        }
    }
}

void ExecutionEngine::backward(ComputationalGraph& graph, int output_operand_id,
                                  const a_tens& loss_gradient,
                                  const std::vector<a_tens>& args) {
    std::unordered_map<int, a_tens> gradient_cache;
    std::unordered_map<int, a_tens> forward_cache;

    // Ensure forward pass is computed
    compute_forward(graph, output_operand_id, forward_cache, args);

    // Start backward pass
    compute_backward(graph, output_operand_id, loss_gradient, forward_cache, gradient_cache);

    // Store gradients in the graph's gradients map
    for (const auto& pair : gradient_cache) {
        graph.gradients[pair.first] = pair.second;
    }
}

void ExecutionEngine::compute_backward(ComputationalGraph& graph, int operand_id,
                                       const a_tens& upstream_gradient,
                                       std::unordered_map<int, a_tens>& forward_cache,
                                       std::unordered_map<int, a_tens>& gradient_cache) {
    if (gradient_cache.find(operand_id) != gradient_cache.end()) {
        // Accumulate gradient
        gradient_cache[operand_id] = AnyTensor(gradient_cache[operand_id].add(upstream_gradient));
        return;
    } else {
        gradient_cache[operand_id] = upstream_gradient;
    }

    Operand& operand = graph.operands[operand_id];

    if (operand.op_type == OperandType::Constant
        || operand.op_type == OperandType::Variable
        || operand.op_type == OperandType::Weight) {
        // Constants & Variables have zero gradient
        // Weights are leaves; gradient is stored
        return;
    }  else {
        OperationRegistry& registry = OperationRegistry::get_instance();
        Operation operation = registry.get_operation(operand.op_type);

        // Get inputs
        std::vector<a_tens> input_tensors;
        for (int input_id : operand.inputs) {
            input_tensors.push_back(forward_cache[input_id]);
        }

        // Get the forward output from the cache
        auto forward_output = forward_cache[operand_id];

        // Compute local gradients
        std::vector<a_tens> local_gradients =
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

void ExecutionEngine::optimize(ComputationalGraph& graph, const a_tens& input_data, const a_tens& target_data, OptimizationArgs& args) {
    // Get optimization arguments
    int batch_size = args.get<int>("batch_size");
    int max_epoch = args.get<int>("max_epoch");
    auto learning_rate = args.get<float>("learning_rate");
    auto loss_function = args.get<LossFunction>("loss_function");
    auto gradient_reg = args.get<GradientRegularizer>("gradient_regularizer");

    AdamOptimizer adam(learning_rate);  // Use Adam optimizer as an example
    int num_batches = std::ceil(static_cast<float>(input_data.shape()[0]) / batch_size);
    std::cout << "NUM_BATCHES: " << num_batches << std::endl;
    for (int epoch = 0; epoch < max_epoch; ++epoch) {
        for (int batch = 0; batch < num_batches; ++batch) {
            // Get batch data
            const a_tens batch_input = input_data.slice({Slice(batch * batch_size, (batch + 1) * batch_size)});
            a_tens batch_target = target_data.slice({Slice(batch * batch_size, (batch + 1) * batch_size)});
            std::cout << "INPUT" << batch_input.get<float>() << std::endl;
            std::cout << "TARGET" << batch_target.get<float>() << std::endl;
            // Forward pass
            a_tens prediction = forward(graph, graph.root_id, {batch_input});

            // Compute loss
            a_tens loss = LossFunctions::compute_loss(prediction, batch_target, loss_function);
            std::cout << "PREDICTION" << prediction.get<float>() << std::endl;
            std::cout << "LOSS" << loss.get<float>() << std::endl;
            // Backward pass
            backward(graph, graph.root_id, loss, {batch_input});

            // Update weights
            for (auto &[operand_id, gradient]: graph.gradients) {
                if (graph.weights.find(operand_id) != graph.weights.end()) {
                    a_tens &weights = graph.weights[operand_id];
                    graph.weights[operand_id] = adam.update(weights, gradient, epoch + 1);
                }
            }
        }
    }
}

} // namespace dio