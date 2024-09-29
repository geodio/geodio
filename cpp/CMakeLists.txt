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
include_directories(src)
include_directories(src/backends)
include_directories(src/operands)
include_directories(src/tensors)

# Create a library target for shared code (tensors, backends, etc.)
add_library(geodio_lib SHARED
    src/tensors/Tensor.cpp
    src/tensors/Tensor.h
    src/tensors/TensorDb.cpp
    src/tensors/TensorDb.h
    src/backends/Backend.h
    src/backends/CPUBackend.h
    src/backends/CPUBackend.cpp
    src/tensors/Tensor_C_API.h
    src/tensors/Tensor_C_API.cpp
    src/backends/BackendManager.h
    src/operands/Operand.h
    src/operands/OperandType.h
    src/operands/ComputationalGraph.h
    src/operands/Operation.h
    src/operands/ExecutionEngine.h
    src/operands/OperationRegistry.h
    src/operands/OperationRegistry.cpp
    src/operands/ExecutionEngine.cpp
    src/operands/operations.cpp
    src/operands/operations.h
    src/tensors/AnyTensor.cpp
    src/tensors/AnyTensor.h
    src/tensors/ApplyType.h
    src/tensors/ITensor.cpp
    src/tensors/ITensor.h
    src/tensors/Slice.h
    src/operands/optimization/OptimizationArgs.cpp
    src/operands/optimization/OptimizationArgs.h
    src/operands/optimization/LossFunction.cpp
    src/operands/optimization/LossFunction.h
    src/operands/optimization/AdamOptimizer.cpp
    src/operands/optimization/AdamOptimizer.h
    src/utils/VectorGenerator.h
    src/operands/Token.h
    src/operands/OperandType.cpp
)

# Create the main executable target for the project
add_executable(geodio
    src/main.cpp
    tests/test_tensors.cpp
    tests/runner.cpp
    tests/test_tensor_db.cpp
    tests/runner.h
    tests/test_tensor_db.h
    tests/test_tensors.h
    tests/test_execution_engine.cpp
    tests/test_execution_engine.h
    tests/test_slice_mem.cpp
    tests/test_slice_mem.h
)

# Link the geodio executable to the geodio_lib
target_link_libraries(geodio PRIVATE geodio_lib)

# Add pybind11 bindings module
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ../../core/cpp_wrappers)

# Ensure the Python interpreter is linked
pybind11_add_module(geodio_bindings src/bindings/geodio_binding.cpp)

# Link the bindings module to the geodio_lib and pybind11
target_link_libraries(geodio_bindings PRIVATE geodio_lib pybind11::module)

# Link the Python3 libraries if necessary (for Python interpreter symbols)
target_link_libraries(geodio_bindings PRIVATE Python3::Python)

# Add tests
enable_testing()
add_test(NAME GeodioTests COMMAND geodio)
