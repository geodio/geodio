// test_tensor_db.cpp

#include "../src/tensors/Tensor.h"
#include "../src/tensors/TensorDb.h"
#include <iostream>

int tensor_db_tests() {
    using namespace dio;

    std::cout << std::endl << "Testing TensorDb functionality..." << std::endl;

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
