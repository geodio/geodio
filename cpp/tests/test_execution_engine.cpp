
#include "test_execution_engine.h"
#include <complex>
#include "../src/operands/OperandType.h"
#include "../src/operands/ComputationalGraph.h"
#include "../src/operands/operations.h"

#include "../src/operands/ExecutionEngine.h"

void test_forward_pass() {
    dio::ComputationalGraph<float> graph;
    dio::initialize_operations<float>();

    // Assign unique IDs to operands
    int x_id = 1;
    int w_id = 2;
    int b_id = 3;
    int wx_id = 4;
    int wx_plus_b_id = 5;
    int y_id = 6;

    // Create operands
    graph.operands[x_id] = dio::Operand(dio::OperandType::Variable, x_id, {});
    graph.operands[w_id] = dio::Operand(dio::OperandType::Variable, w_id, {});
    graph.operands[b_id] = dio::Operand(dio::OperandType::Variable, b_id, {});
    graph.operands[wx_id] = dio::Operand(dio::OperandType::Multiply, wx_id, {w_id, x_id});
    graph.operands[wx_plus_b_id] = dio::Operand(dio::OperandType::Add, wx_plus_b_id, {wx_id, b_id});
    graph.operands[y_id] = dio::Operand(dio::OperandType::Sigmoid, y_id, {wx_plus_b_id});

    // Assign values to variables
    graph.weights[x_id] = std::make_shared<dio::Tensor<float>>(2.0f); // x = 2.0
    graph.weights[w_id] = std::make_shared<dio::Tensor<float>>(3.0f); // w = 3.0
    graph.weights[b_id] = std::make_shared<dio::Tensor<float>>(1.0f); // b = 1.0

    // Expected output: y = sigmoid((w * x) + b) = sigmoid((3 * 2) + 1) = sigmoid(7) ≈ 0.9990889
    float expected_output = 1.0f / (1.0f + std::exp(-7.0f));

    // Perform forward pass
    std::shared_ptr<dio::Tensor<float>> y_output = dio::ExecutionEngine<float>::forward(graph, y_id);

    // Check if the output matches the expected value
    float computed_output = y_output->get_data()[0];
    if (std::abs(computed_output - expected_output) < 1e-6f) {
        std::cout << "Forward pass test passed." << std::endl;
    } else {
        std::cerr << "Forward pass test failed. Expected: " << expected_output
                  << ", Got: " << computed_output << std::endl;
    }
}

void test_backward_pass() {
    dio::ComputationalGraph<float> graph;
    dio::initialize_operations<float>();

    // Assign unique IDs to operands
    int x_id = 1;
    int w_id = 2;
    int b_id = 3;
    int wx_id = 4;
    int wx_plus_b_id = 5;
    int y_id = 6;

    // Create operands (same as before)
    graph.operands[x_id] = dio::Operand(dio::OperandType::Variable, x_id, {});
    graph.operands[w_id] = dio::Operand(dio::OperandType::Variable, w_id, {});
    graph.operands[b_id] = dio::Operand(dio::OperandType::Variable, b_id, {});
    graph.operands[wx_id] = dio::Operand(dio::OperandType::Multiply, wx_id, {w_id, x_id});
    graph.operands[wx_plus_b_id] = dio::Operand(dio::OperandType::Add, wx_plus_b_id, {wx_id, b_id});
    graph.operands[y_id] = dio::Operand(dio::OperandType::Sigmoid, y_id, {wx_plus_b_id});

    // Assign values to variables
    graph.weights[x_id] = std::make_shared<dio::Tensor<float>>(2.0f); // x = 2.0
    graph.weights[w_id] = std::make_shared<dio::Tensor<float>>(3.0f); // w = 3.0
    graph.weights[b_id] = std::make_shared<dio::Tensor<float>>(1.0f); // b = 1.0

    // Perform forward pass
    std::shared_ptr<dio::Tensor<float>> y_output = dio::ExecutionEngine<float>::forward(graph, y_id);

    // Assume loss L = y, so dL/dy = 1
    std::shared_ptr<dio::Tensor<float>> loss_gradient = std::make_shared<dio::Tensor<float>>(1.0f);

    // Perform backward pass
    dio::ExecutionEngine<float>::backward(graph, y_id, loss_gradient);

    // Expected gradients
    float sigmoid_output = y_output->get_data()[0];
    float grad_y = 1.0f;
    float grad_wx_plus_b = grad_y * sigmoid_output * (1.0f - sigmoid_output);
    float grad_wx = grad_wx_plus_b;
    float grad_b = grad_wx_plus_b * 1.0f;
    float grad_w = grad_wx * graph.weights[x_id]->get_data()[0]; // grad_wx * x
    float grad_x = grad_wx * graph.weights[w_id]->get_data()[0]; // grad_wx * w

    // Retrieve computed gradients
    float computed_grad_w = graph.gradients[w_id]->get_data()[0];
    float computed_grad_x = graph.gradients[x_id]->get_data()[0];
    float computed_grad_b = graph.gradients[b_id]->get_data()[0];

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

void execution_engine_tests() {
    std::cout << std::endl << "Testing Execution Engine functionality..." << std::endl;
    test_forward_pass();
    test_backward_pass();
}