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
)
