// main.cpp

#include <iostream>
#include <string>
#include "../tests/runner.h"  // Include the test runner
#include "operands/ExecutionEngine.h"
#include "operands/OperationRegistry.h"
#include "operands/operations.h"
#include "utils/VectorGenerator.h"


int optimize(){
    auto vec_gen = dio::VectorGenerator();
    dio::ComputationalGraph graph;
    dio::initialize_operations();

    int root = 0;
    int x_id = 1;
    int w_id = 2;
    int b_id = 3;
    int l1_id = 4;
    int sig1_id = 5;

    int w2_id = 6;
    int b2_id = 7;
    int l2_id = 8;
    int sig2_id = 9;


    int w3_id = 10;
    int b3_id = 11;
    int l3_id = 12;

    // Create operands
    // Layer 1
    graph.operands[x_id] = dio::Operand(dio::OperandType::Variable, x_id, {});
    graph.var_map[x_id] = 0;
    graph.operands[w_id] = dio::Operand(dio::OperandType::Weight, w_id, {});
    graph.operands[b_id] = dio::Operand(dio::OperandType::Weight, b_id, {});
    graph.operands[l1_id] = dio::Operand(dio::OperandType::LinearTransformation, l1_id, {x_id, w_id, b_id});
    graph.operands[sig1_id] = dio::Operand(dio::OperandType::Sigmoid, sig1_id, {l1_id});
    // Layer 2
    graph.operands[w2_id] = dio::Operand(dio::OperandType::Weight, w2_id, {});
    graph.operands[b2_id] = dio::Operand(dio::OperandType::Weight, b2_id, {});
    graph.operands[l2_id] = dio::Operand(dio::OperandType::LinearTransformation, l2_id, {sig1_id, w2_id, b2_id});
    graph.operands[sig2_id] = dio::Operand(dio::OperandType::Sigmoid, sig2_id, {l2_id});

    // Layer 3
    graph.operands[w3_id] = dio::Operand(dio::OperandType::Weight, w3_id, {});
    graph.operands[b3_id] = dio::Operand(dio::OperandType::Weight, b3_id, {});
    graph.operands[l3_id] = dio::Operand(dio::OperandType::LinearTransformation, l3_id, {sig2_id, w3_id, b3_id});
    graph.operands[root] = dio::Operand(dio::OperandType::Sigmoid, root, {l3_id});

    // Assign values to variables
    std::vector<float> W1 = vec_gen.uniform_11<float>(10);
    std::vector<float> B1 = vec_gen.zeros<float>(5);
    graph.weights[w_id] = dio::make_tensor_ptr<float>(W1, {2, 5});
    graph.weights[b_id] = dio::make_tensor_ptr<float>(B1, {5});

    std::vector<float> W2 = vec_gen.uniform_11<float>(20);
    std::vector<float> B2 = vec_gen.zeros<float>(4);
    graph.weights[w2_id] = dio::make_tensor_ptr<float>(W2, {5, 4});
    graph.weights[b2_id] = dio::make_tensor_ptr<float>(B2, {4});

    std::vector<float> W3 = vec_gen.uniform_11<float>(4);
    std::vector<float> B3 = vec_gen.zeros<float>(1);
    graph.weights[w3_id] = dio::make_tensor_ptr<float>(W3, {4, 1});
    graph.weights[b3_id] = dio::make_tensor_ptr<float>(B3, {1});

    // Define input and target data
    std::vector<float> input_vector({1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,
                                     1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f});
    dio::a_tens input_data = dio::AnyTensor(input_vector, {8, 2});
    std::vector<float> target_vector({0.0f, 0.0f, 1.0f, 1.0f,
                                      0.0f, 0.0f, 1.0f, 1.0f});
    dio::a_tens target_data = dio::AnyTensor(target_vector, {8, 1});

    // Set optimization arguments
    dio::OptimizationArgs args;
    args.set("learning_rate", 0.1f);
    args.set("batch_size", 4);
    args.set("max_epoch", 100);
    args.set("loss_function", dio::LossFunction::MeanSquaredError);
    args.set("gradient_regularizer", dio::GradientRegularizer::Adam);

    // Run optimization
    dio::ExecutionEngine::optimize(graph, input_data, target_data, args);

    return 0;
}

int main(int argc, char* argv[]) {
    // Check command-line arguments
    if (argc > 1) {
        std::string arg = argv[1];
        if (arg == "--run-tests") {
            std::cout << "Running all tests..." << std::endl;
            run_all();
            return 0;
        }
        else if (arg == "--run-tensor-tests") {
            std::cout << "Running tensor tests..." << std::endl;
            run_tensor();
            return 0;
        }
        else if (arg == "--run-db-tests") {
            std::cout << "Running database tests..." << std::endl;
            run_tensor_db();
            return 0;
        }
        else if (arg == "--run-exe-eng-tests") {
            std::cout << "Running execution engine tests..." << std::endl;
            run_execution_engine();
            return 0;
        }
        else if (arg == "--run-slicing") {
            std::cout << "Running execution engine tests..." << std::endl;
            run_slicing();
            return 0;
        }
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] <<
            " [--run-tests | --run-tensor-tests | --run-db-tests | --run-exe-eng-tests | --run-slicing]" << std::endl;
            return 1;
        }
    }

    // If no arguments provided, proceed with normal execution
    // Here, we'll keep the main application code minimal as we'll be building a shared library
    std::cout << "No tests were run. You can use the following options:" << std::endl;
    std::cout << "  --run-tests          Run all tests" << std::endl;
    std::cout << "  --run-tensor-tests   Run tensor tests only" << std::endl;
    std::cout << "  --run-db-tests       Run database tests only" << std::endl;
    std::cout << "  --run-exe-eng-tests  Run execution engine tests only" << std::endl;
    std::cout << "  --run-slicing        Run slicing tests only" << std::endl;

    optimize();
}
