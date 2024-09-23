#include <iostream>
#include <cmath>
#include <vector>
#include <cassert>
#include "../src/operands/OperandType.h"
#include "../src/operands/ComputationalGraph.h"
#include "../src/tensors/Tensor.h"
#include "../src/tensors/AnyTensor.h"
#include "../src/operands/ExecutionEngine.h"


void test_basic_slicing() {
    // Create a 2x3 tensor
    dio::Tensor<float> tensor({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    }, {2, 3});

    // Slice the tensor: select first row (slice(0, 1)) and all columns
    dio::Tensor<float> slice = tensor.slice({dio::Slice(0, 1), dio::Slice(0, 3)});

    // Expected output is [1.0, 2.0, 3.0]
    assert((slice({0, 0})) == 1.0f);
    assert(slice({0, 1}) == 2.0f);
    assert(slice({0, 2}) == 3.0f);

    std::cout << "Basic slicing test passed!" << std::endl;
}

void test_tensor_slicing() {
    std::cout << "Testing Tensor Slicing . . ." << std::endl;

    // Create a 3x3 Tensor
    dio::Tensor<float> tensor({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    }, {3, 3});

    // Create a slice: select rows 1 to 3 and columns 1 to 2 (slice(1,3), slice(1,2))
    dio::Tensor<float> slice = tensor.slice({dio::Slice(1, 3), dio::Slice(1, 3)});

    // Expected sliced data: [[5.0f, 6.0f], [8.0f, 9.0f]]
    std::vector<float> expected_slice = {5.0f, 6.0f, 8.0f, 9.0f};
    bool passed = true;

    // Check values in the slice
    for (size_t i = 0; i < slice.shape()[0]; ++i) {
        for (size_t j = 0; j < slice.shape()[1]; ++j) {
            if (std::abs(slice({i, j}) - expected_slice[i * 2 + j]) > 1e-6f) {
                std::cerr << "Slicing test failed at (" << i << "," << j << "). Expected: "
                          << expected_slice[i * 2 + j] << ", Got: " << slice({i, j}) << std::endl;
                passed = false;
            }
        }
    }

    if (passed) {
        std::cout << "Tensor slicing test passed." << std::endl;
    } else {
        std::cerr << "Tensor slicing test failed." << std::endl;
    }
}

void test_slice_memory_update() {
    std::cout << "Testing Memory Update via Slice . . ." << std::endl;

    // Create a 3x3 Tensor
    dio::Tensor<float> tensor({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    }, {3, 3});

    // Create a slice: select rows 1 to 2 and columns 0 to 2 (slice(1,3), slice(0,3))
    dio::Tensor<float> slice = tensor.slice({dio::Slice(1, 3), dio::Slice(0, 3)});

    // Update the slice values
    slice = 100.0f;

    // Check if the original tensor is updated correctly
    std::vector<float> expected_tensor = {
        1.0f, 2.0f, 3.0f,
        100.0f, 100.0f, 100.0f,
        100.0f, 100.0f, 100.0f
    };

    bool passed = true;

    // Verify the original tensor is updated
    for (size_t i = 0; i < tensor.shape()[0]; ++i) {
        for (size_t j = 0; j < tensor.shape()[1]; ++j) {
            if (std::abs(tensor({i, j}) - expected_tensor[i * 3 + j]) > 1e-6f) {
                std::cerr << "Memory update test failed at (" << i << "," << j << "). Expected: "
                          << expected_tensor[i * 3 + j] << ", Got: " << tensor({i, j}) << std::endl;
                passed = false;
            }
        }
    }

    if (passed) {
        std::cout << "Memory update test passed." << std::endl;
    } else {
        std::cerr << "Memory update test failed." << std::endl;
    }
}

void test_anytensor_slicing() {
    std::cout << "Testing AnyTensor Slicing . . ." << std::endl;

    // Create an AnyTensor wrapping a 3x3 Tensor
    dio::AnyTensor any_tensor = dio::AnyTensor(std::make_shared<dio::Tensor<float>>(std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    }, std::vector<size_t>{3, 3}));

    // Slice the AnyTensor: select rows 0 to 2 and columns 1 to 3
    dio::AnyTensor slice = any_tensor.slice({dio::Slice(0, 2), dio::Slice(1, 3)});

    // Expected slice: [[2.0f, 3.0f], [5.0f, 6.0f]]
    std::vector<float> expected_slice = {2.0f, 3.0f, 5.0f, 6.0f};
    bool passed = true;

    // Check values in the slice
    dio::Tensor<float>& slice_tensor = slice.get<float>();
    for (size_t i = 0; i < slice_tensor.shape()[0]; ++i) {
        for (size_t j = 0; j < slice_tensor.shape()[1]; ++j) {
            if (std::abs(slice_tensor({i, j}) - expected_slice[i * 2 + j]) > 1e-6f) {
                std::cerr << "AnyTensor slicing test failed at (" << i << "," << j << "). Expected: "
                          << expected_slice[i * 2 + j] << ", Got: " << slice_tensor({i, j}) << std::endl;
                passed = false;
            }
        }
    }

    if (passed) {
        std::cout << "AnyTensor slicing test passed." << std::endl;
    } else {
        std::cerr << "AnyTensor slicing test failed." << std::endl;
    }
}

void test_anytensor_memory_update() {
    std::cout << "Testing AnyTensor Memory Update . . ." << std::endl;

    // Create an AnyTensor wrapping a 3x3 Tensor
    dio::AnyTensor any_tensor = dio::AnyTensor(std::make_shared<dio::Tensor<float>>(std::vector<float>{
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    }, std::vector<size_t>{3, 3}));

    // Slice the AnyTensor: select rows 1 to 3 and columns 1 to 3
    dio::AnyTensor slice = any_tensor.slice({dio::Slice(1, 3), dio::Slice(1, 3)});

    // Update the slice values
    slice = dio::AnyTensor(std::make_shared<dio::Tensor<float>>(std::vector<float>{10.0f, 11.0f, 12.0f, 13.0f}, std::vector<size_t>{2, 2}));

    // Expected modified tensor:
    // [[1.0f, 2.0f, 3.0f],
    //  [4.0f, 10.0f, 11.0f],
    //  [7.0f, 12.0f, 13.0f]]
    std::vector<float> expected_tensor = {
        1.0f, 2.0f, 3.0f,
        4.0f, 10.0f, 11.0f,
        7.0f, 12.0f, 13.0f
    };

    bool passed = true;
    dio::Tensor<float>& tensor = any_tensor.get<float>();

    // Verify the original AnyTensor is updated
    for (size_t i = 0; i < tensor.shape()[0]; ++i) {
        for (size_t j = 0; j < tensor.shape()[1]; ++j) {
            if (std::abs(tensor({i, j}) - expected_tensor[i * 3 + j]) > 1e-6f) {
                std::cerr << "AnyTensor memory update test failed at (" << i << "," << j << "). Expected: "
                          << expected_tensor[i * 3 + j] << ", Got: " << tensor({i, j}) << std::endl;
                passed = false;
            }
        }
    }

    if (passed) {
        std::cout << "AnyTensor memory update test passed." << std::endl;
    } else {
        std::cerr << "AnyTensor memory update test failed." << std::endl;
    }
}

void slice_and_update_tests() {
    std::cout << "\nRunning Tensor and AnyTensor slice & memory update tests . . ." << std::endl;
    test_basic_slicing();
    test_tensor_slicing();
    test_slice_memory_update();
    test_anytensor_slicing();
    test_anytensor_memory_update();
    std::cout << "All tests completed." << std::endl;
}
