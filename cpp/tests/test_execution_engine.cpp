// test_execution_engine.h

#include "test_execution_engine.h"
#include <cmath>
#include <iostream>
#include "../src/operands/OperandType.h"
#include "../src/operands/ComputationalGraph.h"
#include "../src/operands/operations.h"
#include "../src/operands/ExecutionEngine.h"

void test_forward_pass() {
    // Assign unique IDs to operands
    int x_id = 1;
    int w_id = 2;
    int b_id = 3;
    int wx_id = 4;
    int wx_plus_b_id = 5;
    int y_id = 6;

    dio::ComputationalGraph graph = dio::ComputationalGraph{y_id};
    dio::initialize_operations();

    // Create operands
    graph.operands[x_id] = dio::Operand(dio::OperandType::Weight, x_id, {});
    graph.operands[w_id] = dio::Operand(dio::OperandType::Weight, w_id, {});
    graph.operands[b_id] = dio::Operand(dio::OperandType::Weight, b_id, {});
    graph.operands[wx_id] = dio::Operand(dio::OperandType::Multiply, wx_id, {w_id, x_id});
    graph.operands[wx_plus_b_id] = dio::Operand(dio::OperandType::Add, wx_plus_b_id, {wx_id, b_id});
    graph.operands[y_id] = dio::Operand(dio::OperandType::Sigmoid, y_id, {wx_plus_b_id});

    // Assign values to variables
    graph.weights[x_id] = dio::make_tensor_ptr<float>(2); // x = 2.0
    graph.weights[w_id] = dio::make_tensor_ptr<float>(3.0f); // w = 3.0
    graph.weights[b_id] = dio::make_tensor_ptr<float>(1.0f); // b = 1.0

    // Expected output: y = sigmoid((w * x) + b) = sigmoid((3 * 2) + 1) = sigmoid(7) ≈ 0.9990889
    float expected_output = 1.0f / (1.0f + std::exp(-7.0f));

    // Perform forward pass
    dio::a_tens y_output = dio::ExecutionEngine::forward(graph, y_id);

    // Check if the output matches the expected value
    float computed_output;

    if (y_output.is<float>()) {
        computed_output = y_output.get<float>().get_data()[0];
    } else {
        std::cerr << "Unexpected data type in y_output" << std::endl;
        return;
    }

    if (std::abs(computed_output - expected_output) < 1e-6f) {
        std::cout << "Forward pass test passed." << std::endl;
    } else {
        std::cerr << "Forward pass test failed. Expected: " << expected_output
                  << ", Got: " << computed_output << std::endl;
    }
}

void test_backward_pass() {
    // Assign unique IDs to operands
    int x_id = 1;
    int w_id = 2;
    int b_id = 3;
    int wx_id = 4;
    int wx_plus_b_id = 5;
    int y_id = 6;

    dio::ComputationalGraph graph = dio::ComputationalGraph{y_id};
    dio::initialize_operations();

    // Create operands (same as before)
    graph.operands[x_id] = dio::Operand(dio::OperandType::Weight, x_id, {});
    graph.operands[w_id] = dio::Operand(dio::OperandType::Weight, w_id, {});
    graph.operands[b_id] = dio::Operand(dio::OperandType::Weight, b_id, {});
    graph.operands[wx_id] = dio::Operand(dio::OperandType::Multiply, wx_id, {w_id, x_id});
    graph.operands[wx_plus_b_id] = dio::Operand(dio::OperandType::Add, wx_plus_b_id, {wx_id, b_id});
    graph.operands[y_id] = dio::Operand(dio::OperandType::Sigmoid, y_id, {wx_plus_b_id});

    // Assign values to variables
    graph.weights[x_id] = dio::make_tensor_ptr<float>(2.0f); // x = 2.0
    graph.weights[w_id] = dio::make_tensor_ptr<float>(3.0f); // w = 3.0
    graph.weights[b_id] = dio::make_tensor_ptr<float>(1.0f); // b = 1.0

    // Perform forward pass
    dio::a_tens y_output = dio::ExecutionEngine::forward(graph, y_id);

    // Assume loss L = y, so dL/dy = 1
    dio::a_tens loss_gradient = dio::make_tensor_ptr<float>(1.0f);

    // Perform backward pass
    dio::ExecutionEngine::backward(graph, y_id, loss_gradient);

    // Expected gradients
    float sigmoid_output = y_output.get<float>().get_data()[0];
    float grad_y = 1.0f;
    float grad_wx_plus_b = grad_y * sigmoid_output * (1.0f - sigmoid_output);
    float grad_wx = grad_wx_plus_b;
    float grad_b = grad_wx_plus_b * 1.0f;
    float x_value = graph.weights[x_id].get<float>().get_data()[0];
    float w_value = graph.weights[w_id].get<float>().get_data()[0];
    float grad_w = grad_wx * x_value; // grad_wx * x
    float grad_x = grad_wx * w_value; // grad_wx * w

    // Retrieve computed gradients
    float computed_grad_w = graph.gradients[w_id].get<float>().get_data()[0];
    float computed_grad_x = graph.gradients[x_id].get<float>().get_data()[0];
    float computed_grad_b = graph.gradients[b_id].get<float>().get_data()[0];

    // Check gradients
    bool passed = true;
    if (std::abs(computed_grad_w - grad_w) > 1e-6f) {
        std::cerr << "Gradient w.r.t w incorrect. Expected: " << grad_w << ", Got: " << computed_grad_w << std::endl;
        passed = false;
    }
    if (std::abs(computed_grad_x - grad_x) > 1e-6f) {
        std::cerr << "Gradient w.r.t x incorrect. Expected: " << grad_x << ", Got: " << computed_grad_x << std::endl;
        passed = false;
    }
    if (std::abs(computed_grad_b - grad_b) > 1e-6f) {
        std::cerr << "Gradient w.r.t b incorrect. Expected: " << grad_b << ", Got: " << computed_grad_b << std::endl;
        passed = false;
    }
    if (passed) {
        std::cout << "Backward pass test passed." << std::endl;
    }
}

void test_conditions_and_jumps() {
    // Initialize computational graph
    dio::ComputationalGraph graph;

    // Operand IDs
    int x_id = 1;
    int y_id = 2;
    int condition_id = 3;
    int true_branch_id = 4;
    int false_branch_id = 5;
    int jump_label_id = 6;
    int jump_id = 7;
    int final_result_id = 8;

    dio::initialize_operations();

    // Create operands
    // x > y ?
    graph.operands[x_id] = dio::Operand(dio::OperandType::Constant, x_id, {});
    graph.operands[y_id] = dio::Operand(dio::OperandType::Constant, y_id, {});
    graph.operands[condition_id] = dio::Operand(dio::OperandType::GreaterThan, condition_id, {x_id, y_id});

    // Branches: True and False
    graph.operands[true_branch_id] = dio::Operand(dio::OperandType::Add, true_branch_id, {x_id, y_id});  // True: x + y
    graph.operands[false_branch_id] = dio::Operand(dio::OperandType::Multiply, false_branch_id, {x_id, y_id});  // False: x * y

    // Jump: If condition is true, jump to the label
    graph.operands[jump_label_id] = dio::Operand(dio::OperandType::Label, jump_label_id, {});
    graph.operands[jump_id] = dio::Operand(dio::OperandType::Jump, jump_id, {jump_label_id});

    // Final result: If the condition is true, the result is x + y; if false, the result is x * y
    graph.operands[final_result_id] = dio::Operand(dio::OperandType::Condition, final_result_id, {condition_id, true_branch_id, false_branch_id});

    // Assign values for variables x and y
    graph.constants[x_id] = dio::make_tensor_ptr<int>(4);  // x = 4
    graph.constants[y_id] = dio::make_tensor_ptr<int>(2);  // y = 2

    // Expected output: since x > y (4 > 2), the result should be true branch (x + y = 4 + 2 = 6)
    int expected_output = 6;

    // Perform forward pass
    dio::a_tens result = dio::ExecutionEngine::forward(graph, final_result_id);

    // Extract the computed result
    int computed_output;
    if (result.is<int>()) {
        computed_output = result.get<int>().get_data()[0];
    } else {
        std::cerr << "Unexpected data type in result" << std::endl;
        return;
    }

    // Verify the output
    if (computed_output == expected_output) {
        std::cout << "Condition and jump test passed." << std::endl;
    } else {
        std::cerr << "Condition and jump test failed. Expected: " << expected_output
                  << ", Got: " << computed_output << std::endl;
    }
}


void execution_engine_tests() {
    std::cout << std::endl << "Testing Execution Engine functionality..." << std::endl;
    std::cout << "Testing Forward . . ." << std::endl;
    test_forward_pass();
    std::cout << "Testing Backward . . ." << std::endl;
    test_backward_pass();
    std::cout << "Testing Jumps . . ." << std::endl;
    test_conditions_and_jumps();
}
