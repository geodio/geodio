// main.cpp

#include <iostream>
#include <string>
#include "../tests/runner.h"  // Include the test runner
#include "operands/ExecutionEngine.h"
#include "operands/OperationRegistry.h"

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
        else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            std::cerr << "Usage: " << argv[0] << " [--run-tests | --run-tensor-tests | --run-db-tests]" << std::endl;
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
}
