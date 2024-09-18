cmake_minimum_required(VERSION 3.29)
project(geodio)

set(CMAKE_CXX_STANDARD 17)

include_directories(cpp/src/tensors)

add_executable(geodio
        cpp/src/tensors/Tensor.cpp
        cpp/src/tensors/Tensor.h
        cpp/src/main.cpp
        cpp/src/tensors/TensorDb.cpp
        cpp/src/tensors/TensorDb.h
        cpp/src/backends/Backend.h
        cpp/src/backends/CPUBackend.h
        cpp/src/backends/CPUBackend.cpp
        cpp/src/tensors/Tensor_C_API.h
        cpp/src/tensors/Tensor_C_API.cpp
        cpp/src/backends/BackendManager.h
        cpp/tests/test_tensors.cpp
        cpp/tests/runner.cpp
        cpp/tests/test_tensor_db.cpp
        cpp/tests/runner.h
        cpp/tests/test_tensor_db.h
        cpp/tests/test_tensors.h
        cpp/src/operands/Operand.h
        cpp/src/operands/OperandType.h
        cpp/src/operands/ComputationalGraph.h
        cpp/src/operands/Operation.h
        cpp/src/operands/ExecutionEngine.h
        cpp/src/operands/OperationRegistry.h
        cpp/src/operands/OperationRegistry.cpp
        cpp/src/operands/ExecutionEngine.cpp
        cpp/src/operands/operations.cpp
        cpp/src/operands/operations.h
        cpp/tests/test_execution_engine.cpp
        cpp/tests/test_execution_engine.h
)
