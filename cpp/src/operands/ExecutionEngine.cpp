//    Geodio Neural Network Framework - A lightweight neural network library focused on custom architecture design and optimization.
//    Copyright (C) Copyright (C) 2024 Geodio (created by Rareș Polenciuc)
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#include "ExecutionEngine.h"
#include "OperationRegistry.h"
#include "Operand.h"
#include "optimization/OptimizationArgs.h"
#include "optimization/AdamOptimizer.h"
#include "optimization/LossFunction.h"
#include "Token.h"
#include "ExecutionContext.h"

namespace dio {

    // Process function calls
    void handleFunctionOperand(ExecutionContext& context, Operand& operand, std::stack<Token>& token_stack) {
        std::vector<int> arg_operand_ids(operand.inputs.begin() + 1, operand.inputs.end());
        for (auto it = arg_operand_ids.rbegin(); it != arg_operand_ids.rend(); ++it) {
            token_stack.emplace(TokenType::Operand, *it);
        }

//        int function_graph_id = operand.inputs[0];
//        ComputationalGraph* function_graph = getGraphById(function_graph_id);
//
//        token_stack.emplace(TokenType::FunctionEnd, operand.id);
//
//        ExecutionContext function_context(function_graph, {}, context.caching_mode, &context);
//        size_t param_index = 0;
//        for (const auto& [param_operand_id, _] : function_graph->var_map) {
//            if (param_index < arg_operand_ids.size()) {
//                int arg_operand_id = arg_operand_ids[param_index];
//                a_tens arg_value = context.get(arg_operand_id);
//                function_context.set(param_operand_id, arg_value);
//                param_index++;
//            } else {
//                throw std::runtime_error("Not enough arguments provided to function");
//            }
//        }
//
//        context_stack.push(function_context);
//        token_stack.emplace(TokenType::FunctionStart, operand.id);
//        token_stack.emplace(TokenType::Operand, function_graph->root_id);
    }

     a_tens ExecutionEngine::execute(ComputationalGraph& graph, int output_operand_id, const std::vector<a_tens>& args = {}) {
        ExecutionContext context {&graph, args, CachingMode::CacheMinimal};
        std::stack<Token> token_stack;
        return compute_forward(context, output_operand_id, token_stack);
    }

    a_tens ExecutionEngine::forward(ComputationalGraph& graph, int output_operand_id, const std::vector<a_tens>& args) {
        ExecutionContext context {&graph, args, CachingMode::CacheAll};
        std::stack<Token> token_stack;
        return compute_forward(context, output_operand_id, token_stack);
    }


    // Main forward computation that uses ExecutionContext
    a_tens ExecutionEngine::compute_forward(ExecutionContext& context, int output_operand_id,  std::stack<Token>& token_stack) {

        token_stack.emplace(TokenType::START);
        token_stack.emplace(TokenType::Operand, output_operand_id);

        while (!token_stack.empty()) {
            Token current_token = token_stack.top();
            token_stack.pop();

            switch (current_token.type) {
                case TokenType::START:
                    break;

                case TokenType::END:
                    context.clean_cache(); // Clean cache when we return to start
                    break;

                case TokenType::Operand:
                    processOperandToken(context, current_token, token_stack);
                    break;

                case TokenType::Return:
                    processReturnToken(context, current_token, token_stack);
                    break;

                case TokenType::Jump:
                    processJumpToken(context, current_token, token_stack);
                    break;

                case TokenType::Label:
                    processLabelToken(context, current_token, token_stack);
                    break;

                case TokenType::ConditionReturn:
                    processConditionReturnToken(context, current_token, token_stack);
                    break;

                case TokenType::ConditionResult:
                    processConditionResultToken(context, current_token, token_stack);
                    break;

                default:
                    throw std::runtime_error("Unknown token type");
            }
        }

        return context.get(output_operand_id);
    }



    void ExecutionEngine::processOperandToken(ExecutionContext& context, const Token &current_token, std::stack<Token>& token_stack) {
        Operand& operand = context.graph->operands[current_token.operand_id];

        // Check if result is already cached for this time step
         if (context.has_operand_result(current_token.operand_id)) {
            return;  // Operand is already cached, skip
        }

        switch (operand.op_type) {
            case OperandType::Constant:
                context.set(current_token.operand_id, context.graph->constants.at(current_token.operand_id));
                break;

            case OperandType::Variable:
                context.set(current_token.operand_id, context.args[context.graph->var_map.at(current_token.operand_id)]);
                break;

            case OperandType::Weight:
                context.set(current_token.operand_id, context.graph->weights.at(current_token.operand_id));
                break;

            case OperandType::Seq:
                // Execute operands in sequence
                for (auto it = operand.inputs.rbegin(); it != operand.inputs.rend(); ++it) {
                    token_stack.emplace(TokenType::Operand, *it);
                }
                break;

            case OperandType::Tick:
                context.current_time_step++;  // Increment time step
                break;

            case OperandType::getTime:
                context.set(current_token.operand_id, a_tens(static_cast<int>(context.current_time_step)));
                break;

            case OperandType::Jump: {
             // Push Jump token
             int label_id = operand.inputs[0];
             // Push Jump token with jump_label
             token_stack.emplace(TokenType::Jump, current_token.operand_id, -1, label_id);
             break;
            }

            case OperandType::Label:
                // Push Label token
                token_stack.emplace(TokenType::Label, current_token.operand_id);
                break;

            case OperandType::Identity:
                token_stack.emplace(TokenType::Return, current_token.operand_id);
                token_stack.emplace(TokenType::Operand, context.get_op(current_token.operand_id).inputs[0]);
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
            case OperandType::FunctionCall:
                //TODO
                 break;
            case OperandType::FunctionPtr:
                //TODO
                 break;
            case OperandType::Function:
                handleFunctionOperand(context, operand, token_stack);
                break;
            case OperandType::String:
                //TODO
                break;
            case OperandType::RETURN:
                token_stack.emplace(TokenType::END, current_token.operand_id);
                token_stack.emplace(TokenType::Return, current_token.operand_id);

                // Push operand returned onto the stack if exists
                if (!operand.inputs.empty()) {
                    token_stack.emplace(TokenType::Operand, operand.inputs[0]);
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

    void ExecutionEngine::processJumpToken(ExecutionContext& context,  const Token &current_token,  std::stack<Token>& token_stack) {
        // The target label ID is stored in jump_label
        int label_id = current_token.jump_label;

        // After the label is processed, we need to store the result of the jump
        // Since jumps might be associated with a result, we can push a Return token
        token_stack.emplace(TokenType::Return, current_token.operand_id);

        // Push the Label token to process the label
        token_stack.emplace(TokenType::Label, label_id);


    }

    void ExecutionEngine::processLabelToken(ExecutionContext& context,  const Token &current_token,  std::stack<Token>& token_stack) {
        Operand &label_operand = context.graph->operands[current_token.operand_id];

        // Push a Return token to process the label after its contents are computed
        token_stack.emplace(TokenType::Return, current_token.operand_id);

        // Push the operands of the label onto the stack to compute them
        for (auto it = label_operand.inputs.rbegin(); it != label_operand.inputs.rend(); ++it) {
            token_stack.emplace(TokenType::Operand, *it);
        }
    }


    void ExecutionEngine::processReturnToken(ExecutionContext& context,  const Token &current_token,  std::stack<Token>& token_stack) {
        // Process the function or operation by retrieving inputs from cache
        Operand &operand = context.graph->operands[current_token.operand_id];

        // Retrieve inputs for the current time step
        std::vector<a_tens> input_tensors = fill_input_tens(operand,
                                                            context.operand_cache,
                                                            context.current_time_step);

        a_tens result;

        // Handle comparison operations separately
        switch (operand.op_type) {
            case OperandType::Label:
                result = input_tensors[input_tensors.size() - 1];
                break;
            case OperandType::Identity:
            case OperandType::Jump:
                result = input_tensors[0];
                break;
            case OperandType::LessThan:
            case OperandType::GreaterThan:
            case OperandType::Equal:
            case OperandType::LessThanOrEqual:
            case OperandType::GreaterThanOrEqual:
                result = evaluate_comparison(operand, input_tensors);
                break;

            case OperandType::SET:
            {
                // Inputs: target ID (input 0), value (input 1)
                int target_id = operand.inputs[0];
                const a_tens& value = input_tensors[0]; // value to set
                Operand& target_operand = context.graph->operands[target_id];

                // Set the value
                switch (target_operand.op_type) {
                    case OperandType::Weight:
                        // Update weights
                        context.graph->weights[target_id] = value;
                        break;
                    default:
                        // Handle other cases
                        context.set(target_id, value);
                        break;
                }

                result = value; // Result of SET can be the value set
            }
                break;
            case OperandType::ACCESS:
            {
                // Inputs: tensor (input 0), indices (input 1..)
                const a_tens& tensor = input_tensors[0];
                std::vector<Slice> slices;
                for (size_t i = 1; i < input_tensors.size(); ++i) {
                    // Assume indices are integers
                    int index = input_tensors[i].get<int>().get_data()[0];
                    slices.emplace_back(index);
                }
                // Access the tensor with slices
                a_tens accessed = tensor.slice(slices);
                result = accessed;
            }
                break;
            case OperandType::RETURN:
            {
                if (!input_tensors.empty()) {
                    result = input_tensors[0];
                } else {
                    result = a_tens(); // Empty tensor
                }
            }
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
        context.set(current_token.operand_id, result);
    }

    void ExecutionEngine::processConditionReturnToken(ExecutionContext& context,  const Token &current_token,  std::stack<Token>& token_stack) {
        // Process the condition
        Operand &condition_operand = context.graph->operands[current_token.operand_id];

        // The condition result is in the cache (after processing the condition expression)
        a_tens condition_result_tensor = context.get(condition_operand.inputs[0]);

        // Evaluate the condition result tensor to a boolean value
        bool cond_met = condition_result_tensor.get<int>().get_data().at(0);

        // Record the result of the condition in the execution path
        context.execution_path[context.current_time_step][condition_operand.id] = cond_met;

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

    void ExecutionEngine::processConditionResultToken(ExecutionContext& context,  const Token &current_token,  std::stack<Token>& token_stack) {
        // Process the condition result
        Operand &condition_operand = context.graph->operands[current_token.operand_id];
        bool cond_met = context.execution_path[context.current_time_step][condition_operand.id];

        // Cache the result of the true or false branch
        if (cond_met) {
            context.set(current_token.operand_id,
                        context.get(condition_operand.inputs[1]));
        } else if (condition_operand.inputs.size() > 2) {
            context.set(current_token.operand_id,
                        context.get(condition_operand.inputs[2]));
        } else {
            // Handle the case where there's no false branch
            throw std::runtime_error("Condition has no false branch");
        }

        // Uncache the condition inputs after use
        context.operand_cache[context.current_time_step].erase(condition_operand.inputs[0]);

        // Uncache the true or false branch inputs if not needed
        int branch_input = cond_met ? condition_operand.inputs[1] : condition_operand.inputs[2];
        context.operand_cache[context.current_time_step].erase(branch_input);

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
        std::unordered_map<int, std::unordered_map<int, a_tens>> gradient_cache;
        int current_time_step = 0;

        // Ensure forward pass is computed
        ExecutionContext context = ExecutionContext(&graph, args, CachingMode::CacheAll);
        compute_forward(context, output_operand_id, token_stack);

        // Start backward pass
        compute_backward(*context.graph, output_operand_id,
                         loss_gradient, context.operand_cache,
                         gradient_cache, &current_time_step, context.execution_path);

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
                for (auto it = operand.inputs.rbegin(); it != operand.  inputs.rend(); ++it) {
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
                if (operand.op_type == OperandType::FunctionCall)
                    handle_function_call();
                else
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

    void ExecutionEngine::handle_function_call() {

    }

} // namespace dio