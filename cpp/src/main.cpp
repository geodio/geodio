#include "tensors/Tensor.h"
#include <iostream>

int main() {
    // Initialize tensor from a nested initializer list
    dio::Tensor<int> tensor1({1, 2, 3, 4});

    std::cout << "Shape: ";
    for (auto dim : tensor1.shape()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    // Save tensor to a file
    tensor1.save_to_file("tensor_data.bin");

    // Load tensor from a file
    dio::Tensor<int> tensor2({});  // Empty tensor
    tensor2.load_from_file("tensor_data.bin");

    std::cout << "Loaded Tensor Shape: ";
    std::cout << tensor2;
    for (auto dim : tensor2.shape()) {
        std::cout << dim << " ";
    }
    std::cout << std::endl;

    return 0;
}
