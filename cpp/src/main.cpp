// main.cpp

#include <iostream>
#include <string>
#include "../tests/runner.h"  // Include the test runner
#include "operands/ExecutionEngine.h"
#include "operands/OperationRegistry.h"
#include "operands/operations.h"


int optimize(){
    dio::ComputationalGraph graph;
    dio::initialize_operations();

    int root = 0;
    int x_id = 1;
    int w_id = 2;
    int b_id = 3;
    int l1_id = 4;

    // Create operands (same as before)
    graph.operands[x_id] = dio::Operand(dio::OperandType::Variable, x_id, {});
    graph.var_map[x_id] = 0;
    graph.operands[w_id] = dio::Operand(dio::OperandType::Weight, w_id, {});
    graph.operands[b_id] = dio::Operand(dio::OperandType::Weight, b_id, {});
    graph.operands[l1_id] = dio::Operand(dio::OperandType::LinearTransformation, l1_id, {x_id, w_id, b_id});
    graph.operands[root] = dio::Operand(dio::OperandType::Sigmoid, root, {l1_id});

    // Assign values to variables
    graph.weights[w_id] = dio::make_tensor_ptr<float>({0.5f, 0.5f}, {2, 1});
    graph.weights[b_id] = dio::make_tensor_ptr<float>({0.0f}, {1});

    // Define input and target data
    std::vector<float> input_vector({1.0f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 0.0f, 0.0f,});
    dio::a_tens input_data = dio::AnyTensor(input_vector, {6, 2});
    std::vector<float> target_vector({1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f});
    dio::a_tens target_data = dio::AnyTensor(target_vector, {6, 1});

    // Set optimization arguments
    dio::OptimizationArgs args;
    args.set("learning_rate", 0.01f);
    args.set("batch_size", 1);
    args.set("max_epoch", 300);
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
