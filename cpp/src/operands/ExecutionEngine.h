#include <stack>
#include "ComputationalGraph.h"
#include "AnyTensor.h"
#include "optimization/OptimizationArgs.h"
#include "Token.h"
#include "Operation.h"

#ifndef GEODIO_EXECUTIONENGINE_H
#define GEODIO_EXECUTIONENGINE_H

namespace dio {
class ExecutionEngine {
public:

    static a_tens forward(ComputationalGraph& graph, int output_operand_id, const std::vector<a_tens>& args = {});
    static bool evaluate_condition(ComputationalGraph& graph, Operand& operand,
                                             std::vector<a_tens> inputs);
    static void backward(ComputationalGraph& graph,
                         int output_operand_id,
                         const a_tens& loss_gradient,
                         const std::vector<a_tens>& args = {});

    static void optimize(ComputationalGraph &graph, const a_tens &input_data, const a_tens &target_data,
             OptimizationArgs &args);

private:
    static void compute_backward(ComputationalGraph& graph, int output_operand_id,
                                       const a_tens& loss_gradient,
                                       std::unordered_map<int, std::unordered_map<int, a_tens>>& forward_cache,
                                       std::unordered_map<int, std::unordered_map<int, a_tens>>& gradient_cache,
                                       int* current_time_step,
                                       std::unordered_map<int, std::unordered_map<int, bool>>& execution_path);

    static a_tens compute_forward(ComputationalGraph &graph, int output_operand_id, const std::vector<a_tens> &args,
                                  std::stack<Token> &token_stack,
                                  std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                                  int* current_time_step,
                                  std::unordered_map<int, std::unordered_map<int, bool>>& execution_path);

    static void handle_regular(const std::unordered_map<int, std::unordered_map<int, a_tens>> &forward_cache,
                                    const int *current_time_step, std::stack<std::tuple<int, a_tens>> &operand_stack,
                                    int operand_id, const dio::tensor &upstream_gradient,
                                    Operand &operand);

    static  std::vector<a_tens> fill_input_tens(const Operand &operand, const std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                    int current_time_step);

    static void processOperandToken(ComputationalGraph &graph, const Token &current_token, const std::vector<a_tens> &args,
                             std::stack<Token> &token_stack,
                             std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                             int *current_time_step);

    static void processReturnToken(ComputationalGraph &graph, const Token &current_token,
                            std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                            int current_time_step);

    static void processConditionReturnToken(ComputationalGraph &graph, const Token &current_token,
                                     std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                                     int current_time_step, std::stack<Token> &token_stack,
                                     std::unordered_map<int, std::unordered_map<int, bool>> &execution_path);

    static void processConditionResultToken(ComputationalGraph &graph, const Token &current_token,
                                     std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                                     int current_time_step,
                                     std::unordered_map<int, std::unordered_map<int, bool>> &execution_path);

    static a_tens evaluate_comparison(const Operand &operand, const std::vector<a_tens> &inputs);

    static void processJumpToken(ComputationalGraph &graph, const Token &current_token, std::stack<Token> &token_stack,
                          std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                          int current_time_step);

    static void processLabelToken(ComputationalGraph &graph, const Token &current_token, std::stack<Token> &token_stack,
                           std::unordered_map<int, std::unordered_map<int, a_tens>> &time_aware_cache,
                           int current_time_step);
};
} // namespace dio

#endif //GEODIO_EXECUTIONENGINE_H
