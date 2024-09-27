#include "ExecutionEngine.h"
#include "OperationRegistry.h"
#include "Operand.h"
#include "optimization/OptimizationArgs.h"
#include "optimization/AdamOptimizer.h"
#include "optimization/LossFunction.h"
#include "Token.h"

namespace dio {

    a_tens get_tensor(ComputationalGraph& graph,
                      int tensor_id,
                      int current_time_step,
                      std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                      const std::vector<a_tens>& args) {
        auto operand = graph.operands.at(tensor_id);
        switch (operand.op_type) {
            case OperandType::Constant:
                return graph.constants.at(tensor_id);

            case OperandType::Weight:
                return graph.weights.at(tensor_id);

            case OperandType::Variable:
                return args[graph.var_map.at(tensor_id)];

            default:
                return time_aware_cache[current_time_step][tensor_id];
        }
    }

    a_tens ExecutionEngine::forward(ComputationalGraph& graph, int output_operand_id, const std::vector<a_tens>& args) {
        std::stack<Token> token_stack;
        std::unordered_map<int, a_tens> return_stack;
        std::unordered_map<int, std::unordered_map<int, a_tens>> time_aware_cache;
        std::unordered_map<int, std::unordered_map<int, bool>> execution_path;
        int current_time_step = 0;

        return compute_forward(graph, output_operand_id, args, token_stack,
                               time_aware_cache, &current_time_step, execution_path);

    }

    a_tens ExecutionEngine::compute_forward(ComputationalGraph &graph,
                                            int output_operand_id,
                                            const std::vector<a_tens> &args,
                                            std::stack<Token> &token_stack,
                                            std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                                            int* current_time_step,
                                            std::unordered_map<int, std::unordered_map<int, bool>>& execution_path) {
        token_stack.emplace(TokenType::Operand, output_operand_id);  // Start with the output operand

        while (!token_stack.empty()) {
            Token current_token = token_stack.top();
            token_stack.pop();

            switch (current_token.type) {
                case TokenType::Operand:
                    processOperandToken(graph, current_token, args, token_stack, time_aware_cache, current_time_step);
                    break;

                case TokenType::Return:
                    processReturnToken(graph, current_token, time_aware_cache, *current_time_step);
                    break;

                case TokenType::ConditionReturn:
                    processConditionReturnToken(graph, current_token, time_aware_cache, *current_time_step, token_stack, execution_path);
                    break;

                case TokenType::ConditionResult:
                    processConditionResultToken(graph, current_token, time_aware_cache, *current_time_step, execution_path);
                    break;

                default:
                    throw std::runtime_error("Unknown token type");
            }
        }

        // Return the final result at the last time step
        return time_aware_cache[*current_time_step][output_operand_id];
    }

    void ExecutionEngine::processOperandToken(ComputationalGraph &graph,
                                              const Token &current_token,
                                              const std::vector<a_tens> &args,
                                              std::stack<Token> &token_stack,
                                              std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                                              int* current_time_step) {
        Operand &operand = graph.operands[current_token.operand_id];

        // Check if result is already cached for this time step
        if (time_aware_cache[*current_time_step].find(current_token.operand_id) != time_aware_cache[*current_time_step].end()) {
            return;  // Skip if already computed
        }

        switch (operand.op_type) {
            case OperandType::Constant:
                time_aware_cache[*current_time_step][current_token.operand_id] = graph.constants.at(current_token.operand_id);
                break;

            case OperandType::Weight:
                time_aware_cache[*current_time_step][current_token.operand_id] = graph.weights.at(current_token.operand_id);
                break;

            case OperandType::Variable:
                time_aware_cache[*current_time_step][current_token.operand_id] = args[graph.var_map.at(current_token.operand_id)];
                break;

            case OperandType::Seq:
                // Execute operands in sequence
                for (auto it = operand.inputs.rbegin(); it != operand.inputs.rend(); ++it) {
                    token_stack.emplace(TokenType::Operand, *it);
                }
                break;

            case OperandType::Tick:
                (*current_time_step)++;  // Increment time step
                break;

            case OperandType::getTime:
                time_aware_cache[*current_time_step][current_token.operand_id] = a_tens(static_cast<int>(*current_time_step));
                break;

            case OperandType::Jump:
                token_stack.emplace(TokenType::Operand, operand.inputs[0]);  // Jump to label
                break;

            case OperandType::Label:
                // Do nothing for labels
                break;

            case OperandType::Condition:
                // Handle condition
                token_stack.emplace(TokenType::ConditionReturn, current_token.operand_id);

                // Push the condition expression operand
                token_stack.emplace(TokenType::Operand, operand.inputs[0]);
                break;

            case OperandType::LessThan:
            case OperandType::GreaterThan:
            case OperandType::Equal:
            case OperandType::LessThanOrEqual:
            case OperandType::GreaterThanOrEqual:
                // Push a Return token to evaluate the comparison after inputs are processed
                token_stack.emplace(TokenType::Return, current_token.operand_id);

                // Push all input operands onto the stack to evaluate them first
                for (auto it = operand.inputs.rbegin(); it != operand.inputs.rend(); ++it) {
                    token_stack.emplace(TokenType::Operand, *it);
                }
                break;

            default:
                // For other operations, push a Return token and operands
                token_stack.emplace(TokenType::Return, current_token.operand_id);

                // Push operands onto the stack
                for (auto it = operand.inputs.rbegin(); it != operand.inputs.rend(); ++it) {
                    token_stack.emplace(TokenType::Operand, *it);
                }
                break;
        }
    }

    void ExecutionEngine::processReturnToken(ComputationalGraph &graph,
                                             const Token &current_token,
                                             std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                                             int current_time_step) {
        // Process the function or operation by retrieving inputs from cache
        Operand &operand = graph.operands[current_token.operand_id];

        // Retrieve inputs for the current time step
        std::vector<a_tens> input_tensors = fill_input_tens(operand, time_aware_cache, current_time_step);

        a_tens result;

        // Handle comparison operations separately
        switch (operand.op_type) {
            case OperandType::LessThan:
            case OperandType::GreaterThan:
            case OperandType::Equal:
            case OperandType::LessThanOrEqual:
            case OperandType::GreaterThanOrEqual:
                result = evaluate_comparison(operand, input_tensors);
                break;

            default: {
                // Retrieve the operation and compute the result
                OperationRegistry &registry = OperationRegistry::get_instance();
                Operation operation = registry.get_operation(operand.op_type);
                result = operation.forward(input_tensors);
            }
                break;
        }

        // Cache the result of the operation
        time_aware_cache[current_time_step][current_token.operand_id] = result;
    }

    void ExecutionEngine::processConditionReturnToken(ComputationalGraph &graph,
                                                      const Token &current_token,
                                                      std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                                                      int current_time_step,
                                                      std::stack<Token> &token_stack,
                                                      std::unordered_map<int, std::unordered_map<int, bool>>& execution_path) {
        // Process the condition
        Operand &condition_operand = graph.operands[current_token.operand_id];

        // The condition result is in the cache (after processing the condition expression)
        a_tens condition_result_tensor = time_aware_cache[current_time_step][condition_operand.inputs[0]];

        // Evaluate the condition result tensor to a boolean value
        bool cond_met = condition_result_tensor.get<int>().get_data().at(0);

        // Record the result of the condition in the execution path
        execution_path[current_time_step][condition_operand.id] = cond_met;

        // Push the ConditionResult token
        token_stack.emplace(TokenType::ConditionResult, current_token.operand_id);

        // Push the appropriate branch onto the stack
        if (cond_met) {
            // True branch
            token_stack.emplace(TokenType::Operand, condition_operand.inputs[1]);
        } else if (condition_operand.inputs.size() > 2) {
            // False branch
            token_stack.emplace(TokenType::Operand, condition_operand.inputs[2]);
        } else {
            // No false branch, handle accordingly
        }
    }

    void ExecutionEngine::processConditionResultToken(ComputationalGraph &graph,
                                                      const Token &current_token,
                                                      std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                                                      int current_time_step,
                                                      std::unordered_map<int, std::unordered_map<int, bool>>& execution_path) {
        // Process the condition result
        Operand &condition_operand = graph.operands[current_token.operand_id];
        bool cond_met = execution_path[current_time_step][condition_operand.id];

        // Cache the result of the true or false branch
        if (cond_met) {
            time_aware_cache[current_time_step][current_token.operand_id] =
                time_aware_cache[current_time_step][condition_operand.inputs[1]];
        } else if (condition_operand.inputs.size() > 2) {
            time_aware_cache[current_time_step][current_token.operand_id] =
                time_aware_cache[current_time_step][condition_operand.inputs[2]];
        } else {
            // Handle the case where there's no false branch
            throw std::runtime_error("Condition has no false branch");
        }
    }

    std::vector<a_tens> ExecutionEngine::fill_input_tens(const Operand &operand,
                                      const std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                                      int current_time_step) {
        std::vector<a_tens> input_tensors;
        input_tensors.reserve(operand.inputs.size());
        for (int input_id: operand.inputs)
                input_tensors.push_back(time_aware_cache.at(current_time_step).at(input_id));
        return input_tensors;
    }

    // Macro to define the comparison logic based on the OperandType
    #define APPLY_ELEMENTWISE_COMPARISON(T, V, op_type, left_tensor, right_tensor, result) \
        switch (op_type) { \
            case OperandType::LessThan: \
                std::transform(left_tensor.begin(), left_tensor.end(), right_tensor.begin(), result.begin(), \
                               [](T a, V b) { return a < b; }); \
                break; \
            case OperandType::GreaterThan: \
                std::transform(left_tensor.begin(), left_tensor.end(), right_tensor.begin(), result.begin(), \
                               [](T a, V b) { return a > b; }); \
                break; \
            case OperandType::Equal: \
                std::transform(left_tensor.begin(), left_tensor.end(), right_tensor.begin(), result.begin(), \
                               [](T a, V b) { return a == b; }); \
                break; \
            case OperandType::LessThanOrEqual: \
                std::transform(left_tensor.begin(), left_tensor.end(), right_tensor.begin(), result.begin(), \
                               [](T a, V b) { return a <= b; }); \
                break; \
            case OperandType::GreaterThanOrEqual: \
                std::transform(left_tensor.begin(), left_tensor.end(), right_tensor.begin(), result.begin(), \
                               [](T a, V b) { return a >= b; }); \
                break; \
            default: \
                throw std::runtime_error("Unsupported condition type"); \
        }

    template <typename T, typename V>
    bool evaluate_condition_internal(const a_tens& left_value, const a_tens& right_value, OperandType op_type) {
        bool condition_met = false;

        // Instead of directly accessing data, use access functions that support views
        auto left_tensor = left_value.get<T>();  // Access as tensor view
        auto right_tensor = right_value.get<V>();  // Access as tensor view

        // Ensure both tensors have the same size for elementwise comparison
        if (left_tensor.size() != right_tensor.size()) {
            throw std::runtime_error("Mismatched tensor sizes for condition evaluation");
        }

        // Create a result vector to store the comparison results
        std::vector<bool> result(left_tensor.size());

        // Apply elementwise comparison using the macro
        APPLY_ELEMENTWISE_COMPARISON(T, V, op_type, left_tensor, right_tensor, result)

        // Check if all elements meet the condition
        condition_met = std::all_of(result.begin(), result.end(), [](bool x) { return x; });

        return condition_met;
    }

    #define EVALUATE_CONDITION_FOR_TYPE(T, V) \
        if (left_value.is<T>() && right_value.is<V>()) { \
            return evaluate_condition_internal<T, V>(left_value, right_value, operand.op_type); \
        }

    #define EVALUATE_CONDITION_FOR_ALL_TYPES(T1, T2) \
        EVALUATE_CONDITION_FOR_TYPE(T1, T2) \
        EVALUATE_CONDITION_FOR_TYPE(T2, T1) \
        EVALUATE_CONDITION_FOR_TYPE(T1, T1) \
        EVALUATE_CONDITION_FOR_TYPE(T2, T2)

    // Expanded for all type pairs
    #define EVALUATE_ALL_CONDITIONS() \
            EVALUATE_CONDITION_FOR_ALL_TYPES(float, int) \
            EVALUATE_CONDITION_FOR_ALL_TYPES(float, double) \
            EVALUATE_CONDITION_FOR_ALL_TYPES(int, double)

    bool ExecutionEngine::evaluate_condition(ComputationalGraph& graph, Operand& operand, std::vector<a_tens> inputs) {
        // Retrieve values for the condition's operands
        const a_tens& left_value = inputs[0];  // First operand
        const a_tens& right_value = inputs[1];  // Second operand

        // Macro for type checking and function invocation
        EVALUATE_ALL_CONDITIONS()

        // If none of the types match
        throw std::runtime_error("Unsupported tensor types for condition evaluation");
    }

    bool evaluate_condition_internal_untyped(const a_tens& left_value, const a_tens& right_value,const Operand& operand) {
        EVALUATE_ALL_CONDITIONS()

        // If none of the types match
        throw std::runtime_error("Unsupported tensor types for condition evaluation");
    }

    // New function to handle comparison operations
    a_tens ExecutionEngine::evaluate_comparison(const Operand &operand,
                                                const std::vector<a_tens> &inputs) {
        // Retrieve values for the comparison operands
        const a_tens &left_value = inputs[0];  // First operand
        const a_tens &right_value = inputs[1]; // Second operand

        // Use your existing evaluate_condition_internal function
        int result = evaluate_condition_internal_untyped(left_value, right_value, operand);

        // Convert the boolean result to a tensor (implement this in your a_tens class)
        auto r = make_tensor_ptr(result);
        if (!r.is<int>())
            std::cout << "PROBLEM" << std::endl;
        return r;
    }



    void ExecutionEngine::backward(ComputationalGraph& graph, int output_operand_id,
                                      const a_tens& loss_gradient,
                                      const std::vector<a_tens>& args) {
        std::stack<Token> token_stack;
        std::unordered_map<int, a_tens> return_stack;
        std::unordered_map<int, std::unordered_map<int, a_tens>> forward_cache;
        std::unordered_map<int, std::unordered_map<int, a_tens>> gradient_cache;
        int current_time_step = 0;
        std::unordered_map<int, std::unordered_map<int, bool>> execution_path;

        // Ensure forward pass is computed
        compute_forward(graph, output_operand_id, args, token_stack, forward_cache, &current_time_step, execution_path);

        // Start backward pass
        compute_backward(graph, output_operand_id,
                         loss_gradient, forward_cache,
                         gradient_cache, &current_time_step, execution_path);

        // Store gradients in the graph's gradients map
        for (const auto& [time_step, gradients_at_time] : gradient_cache) {
            for (const auto& [operand_id, gradient] : gradients_at_time) {
                graph.gradients[operand_id] = gradient; // TODO
            }
        }
    }

    void ExecutionEngine::compute_backward(ComputationalGraph& graph, int output_operand_id,
                                           const a_tens& loss_gradient,
                                           std::unordered_map<int, std::unordered_map<int, a_tens>>& forward_cache,
                                           std::unordered_map<int, std::unordered_map<int, a_tens>>& gradient_cache,
                                           int* current_time_step,
                                           std::unordered_map<int, std::unordered_map<int, bool>>& execution_path) {
        std::stack<std::tuple<int, a_tens>> operand_stack;  // Stack for processing operands and gradients
        operand_stack.emplace(output_operand_id, loss_gradient);  // Start with the output operand

        while (!operand_stack.empty() && (*current_time_step) > -1) {
            auto [operand_id, upstream_gradient] = operand_stack.top();
            operand_stack.pop();

            // Check if the gradient for this operand and time step is already cached
            if (gradient_cache[*current_time_step].find(operand_id) != gradient_cache[*current_time_step].end()) {
                // Accumulate gradient for the current time step
                gradient_cache[*current_time_step][operand_id] = gradient_cache[*current_time_step][operand_id].add(
                        upstream_gradient);
                continue;
            } else
                // Cache upstream gradient for the current time step
                gradient_cache[*current_time_step][operand_id] = upstream_gradient;


            Operand& operand = graph.operands[operand_id];
            // Leaf nodes like constants, variables, and weights do not propagate gradients
            if (operand.op_type == OperandType::Constant
                || operand.op_type == OperandType::Variable
                || operand.op_type == OperandType::Weight
                || operand.op_type == OperandType::Label) {
                // Labels are flow control points and don't require gradient computation, just flow continuation
                continue;
            } else if (operand.op_type == OperandType::Jump) {
                // 'Jumps' don't have gradients, but we must propagate to the target label
                int jump_target = operand.inputs[0];  // Assuming the first input is the label ID
                operand_stack.emplace(jump_target, upstream_gradient);
            } else if (operand.op_type == OperandType::Seq) {
                // Process operands sequentially
                for (auto it = operand.inputs.rbegin(); it != operand.inputs.rend(); ++it) {
                    operand_stack.emplace(*it, upstream_gradient);
                }
            } else if (operand.op_type == OperandType::Tick) {
                // Subtract time when encountering a Tick
                (*current_time_step)--;
                continue;
            } else if (operand.op_type == OperandType::Condition) {
                // Conditions don't have gradients but propagate to the correct branch
                bool took_true_branch = execution_path[*current_time_step][operand.id];

                if (took_true_branch)
                    // Propagate gradient to the true branch
                    operand_stack.emplace(operand.inputs[2], upstream_gradient);
                else if (operand.inputs.size() > 3)
                    // If a false branch exists, propagate gradient to the false branch
                    operand_stack.emplace(operand.inputs[3], upstream_gradient);
            } else
                handle_regular(forward_cache,
                               current_time_step, operand_stack, operand_id,
                               upstream_gradient, operand);

        }
    }

    // Handle regular operations with gradients
    void ExecutionEngine::handle_regular(const std::unordered_map<int, std::unordered_map<int, a_tens>> &forward_cache,
                                    const int *current_time_step, std::stack<std::tuple<int, a_tens>> &operand_stack,
                                    int operand_id, const dio::tensor &upstream_gradient,
                                    Operand &operand) {
        OperationRegistry& registry = OperationRegistry::get_instance();
        Operation operation = registry.get_operation(operand.op_type);

        // Retrieve input tensors for the current time step
        std::vector<a_tens> input_tensors = fill_input_tens(operand, forward_cache, *current_time_step);

        // Get the forward output from the cache for the current time step
        a_tens forward_output = forward_cache.at(*current_time_step).at(operand_id);

        // Compute local gradients using the operation's backward function
        std::vector<a_tens> local_gradients = operation.backward(input_tensors, upstream_gradient, forward_output);

        // Push each input operand and its gradient onto the stack for further processing
        for (size_t i = 0; i < operand.inputs.size(); ++i)
            operand_stack.emplace(operand.inputs[i], local_gradients[i]);
    }


    void ExecutionEngine::optimize(ComputationalGraph& graph, const a_tens& input_data,
                                   const a_tens& target_data, OptimizationArgs& args) {
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
                std::cout << "INPUT: " << batch_input.get<float>() << std::endl;
                std::cout << "    TARGET: " << batch_target.get<float>() << std::endl;
                // Forward pass
                a_tens prediction = forward(graph, graph.root_id, {batch_input});

                // Compute loss
                a_tens loss = LossFunctions::compute_loss(prediction, batch_target, loss_function);
                a_tens loss_grad = LossFunctions::compute_loss_gradient(prediction, batch_target, loss_function);
                std::cout << "    PREDICTION: " << prediction.get<float>() << std::endl;
                std::cout << "    LOSS: " << loss.get<float>() << std::endl;
                // Backward pass
                backward(graph, graph.root_id, loss_grad, {batch_input});

                // Update weights
                for (auto &[operand_id, gradient]: graph.gradients) {
                    if (graph.weights.find(operand_id) != graph.weights.end()) {
                        a_tens &weights = graph.weights[operand_id];
                        graph.weights[operand_id] = adam.update(weights, gradient, epoch + 1);
                    }
                }
            }
            std::cout << std::endl;
        }
    }

} // namespace dio