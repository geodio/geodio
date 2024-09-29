cmake_minimum_required(VERSION 3.29)
project(geodio)

set(CMAKE_CXX_STANDARD 17)

# Fetch pybind11 via FetchContent
include(FetchContent)
FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.10.1
)
FetchContent_MakeAvailable(pybind11)

# Find Python interpreter and development package
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
message(STATUS "Using Python executable: ${Python3_EXECUTABLE}")
message(STATUS "Python include dirs: ${Python3_INCLUDE_DIRS}")

# Include directories for your project
include_directories(cpp/src)
include_directories(cpp/src/backends)
include_directories(cpp/src/operands)
include_directories(cpp/src/tensors)

# Create a library target for shared code (tensors, backends, etc.)
add_library(geodio_lib SHARED
    cpp/src/tensors/Tensor.cpp
    cpp/src/tensors/Tensor.h
    cpp/src/tensors/TensorDb.cpp
    cpp/src/tensors/TensorDb.h
    cpp/src/backends/Backend.h
    cpp/src/backends/CPUBackend.h
    cpp/src/backends/CPUBackend.cpp
    cpp/src/tensors/Tensor_C_API.h
    cpp/src/tensors/Tensor_C_API.cpp
    cpp/src/backends/BackendManager.h
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
    cpp/src/tensors/AnyTensor.cpp
    cpp/src/tensors/AnyTensor.h
    cpp/src/tensors/ApplyType.h
    cpp/src/tensors/ITensor.cpp
    cpp/src/tensors/ITensor.h
    cpp/src/tensors/Slice.h
    cpp/src/operands/optimization/OptimizationArgs.cpp
    cpp/src/operands/optimization/OptimizationArgs.h
    cpp/src/operands/optimization/LossFunction.cpp
    cpp/src/operands/optimization/LossFunction.h
    cpp/src/operands/optimization/AdamOptimizer.cpp
    cpp/src/operands/optimization/AdamOptimizer.h
    cpp/src/utils/VectorGenerator.h
    cpp/src/operands/Token.h
    cpp/src/operands/OperandType.cpp
)

# Create the main executable target for the project
add_executable(geodio
    cpp/src/main.cpp
    cpp/tests/test_tensors.cpp
    cpp/tests/runner.cpp
    cpp/tests/test_tensor_db.cpp
    cpp/tests/runner.h
    cpp/tests/test_tensor_db.h
    cpp/tests/test_tensors.h
    cpp/tests/test_execution_engine.cpp
    cpp/tests/test_execution_engine.h
    cpp/tests/test_slice_mem.cpp
    cpp/tests/test_slice_mem.h
)

# Link the geodio executable to the geodio_lib
target_link_libraries(geodio PRIVATE geodio_lib)

# Add pybind11 bindings module
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../core/cpp_wrappers)

# Ensure the Python interpreter is linked
pybind11_add_module(tensor_bindings cpp/src/bindings/a_tens_binding.cpp)

# Link the bindings module to the geodio_lib and pybind11
target_link_libraries(tensor_bindings PRIVATE geodio_lib pybind11::module)

# Link the Python3 libraries if necessary (for Python interpreter symbols)
target_link_libraries(tensor_bindings PRIVATE Python3::Python)

# Add tests
enable_testing()
add_test(NAME GeodioTests COMMAND geodio)
