# Prevent in-source builds
if(${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
    message(FATAL_ERROR "In-source builds are not allowed. Please run CMake from a separate 'build' directory.")
endif()

cmake_minimum_required(VERSION 3.29)
project(geodio VERSION 0.1)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Get the parent directory of the source directory
get_filename_component(PARENT_DIR ${CMAKE_SOURCE_DIR} DIRECTORY)

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Now this will point to the correct 'python/geodio' folder in the parent directory
set(PYTHON_BINDINGS_OUTPUT_DIR ${PARENT_DIR}/python/geodio)

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
include_directories(tests)

# Create a shared library for geodio_core (tensors, backends, etc.)
add_library(geodio_core SHARED
    src/tensors/Tensor.cpp
    src/tensors/TensorDb.cpp
    src/backends/CPUBackend.cpp
    src/tensors/Tensor_C_API.cpp
    src/operands/OperationRegistry.cpp
    src/operands/ExecutionEngine.cpp
    src/operands/operations.cpp
    src/tensors/AnyTensor.cpp
    src/operands/optimization/OptimizationArgs.cpp
    src/operands/optimization/LossFunction.cpp
    src/operands/optimization/AdamOptimizer.cpp
    src/operands/OperandType.cpp
)

target_include_directories(geodio_core PUBLIC src)

# Create the main executable target for the project
add_executable(geodio_main
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

# Link the executable to the core library
target_link_libraries(geodio_main PRIVATE geodio_core)

# Add pybind11 bindings module
pybind11_add_module(geodio_bindings src/bindings/geodio_binding.cpp)

# Link the bindings module to the geodio_core library
target_link_libraries(geodio_bindings PRIVATE geodio_core pybind11::module Python3::Python)

# Ensure Python bindings are placed in the Python package directory (geodio)
set_target_properties(geodio_bindings PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PYTHON_BINDINGS_OUTPUT_DIR})

# Add installation rules to install the Python bindings into the 'python' directory
install(TARGETS geodio_bindings DESTINATION ${PYTHON_BINDINGS_OUTPUT_DIR})
install(TARGETS geodio_core DESTINATION ${CMAKE_BINARY_DIR}/lib)
install(FILES src/tensors/Tensor.h DESTINATION include)

# Add tests
enable_testing()
add_test(NAME GeodioTests COMMAND geodio_main)
