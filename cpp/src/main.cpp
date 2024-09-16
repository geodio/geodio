#include "tensors/Tensor.h"
#include "TensorDb.h"
#include <iostream>


int tensor_db() {
    using namespace dio;

    std::cout << std::endl << "Testing tensorDb functionality. . ." << std::endl;

    // Create an empty TensorDb
    TensorDb<float> db;

    // Create some tensors
    Tensor<float> tensor1({1.0f, 2.0f, 3.0f}, {3});
    Tensor<float> tensor2({4.3f, 5.0f, 6.0f, 7.0f}, {2, 2});

    // Add tensors to the database
    db.add_tensor(1, tensor1);
    db.add_tensor(2, tensor2);

    // Retrieve a tensor
    auto retrieved_tensor = db.get_tensor(1);
    std::cout << "Retrieved Tensor 1: " << *retrieved_tensor << std::endl;

    // Save the database to a file
    db.save("tensor_db.bin");

    // Load the database from a file
    TensorDb<float> db_loaded("tensor_db.bin");
    auto loaded_tensor = db_loaded.get_tensor(2);
    std::cout << "Loaded Tensor 2: " << *loaded_tensor << std::endl;

    // Asynchronous loading
    auto future_tensor = db_loaded.get_tensor_async(1);
    auto async_tensor = future_tensor.get();
    std::cout << "Asynchronously Loaded Tensor 1: " << *async_tensor << std::endl;

    return 0;
}


int main() {

    // Initialize a 2D tensor with float literals
    dio::Tensor<float> tensor2D {{1.0f, 2.0f, 3.0f, 4.0f}, {2, 2}};
    std::cout << "2D Tensor: " << tensor2D << std::endl;

    // Access elements
    std::cout << "Element at (0, 1): " << tensor2D({0, 1}) << std::endl;

    // Initialize a 3D tensor
    dio::Tensor<float> tensor3D  {
            {1.0f, 2.0f,
            3.0f, 4.0f,
            5.0f, 6.0f,
            7.0f, 8.0f},
            {2, 2, 2}
    };
    std::cout << "3D Tensor: " << tensor3D << std::endl;

    // Access elements
    std::cout << "Element at (1, 0, 1): " << tensor3D({1, 0, 1}) << std::endl;

    // Initialize a scalar tensor
    auto scalar_tensor = dio::Tensor<float>(42.0f);
    std::cout << "Scalar Tensor: " << scalar_tensor << std::endl;

    dio::Tensor<float> result = scalar_tensor * tensor3D;

    std::cout << "Multiplication result: " << result <<std::endl;

    tensor_db();
    return 0;
}
