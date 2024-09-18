#include <iostream>
#include <vector>
#include <cmath>
#include "../src/tensors/Tensor.h"

void test_default_constructor() {
    dio::Tensor<int> tensor;
    if (tensor.num_dimensions() != 0) {
        std::cerr << "Default constructor test failed!" << std::endl;
    } else {
        std::cout << "Default constructor test passed." << std::endl;
    }
}

void test_scalar_constructor() {
    dio::Tensor<int> tensor(5);
    if (!tensor.is_scalar() || tensor.get_data()[0] != 5) {
        std::cerr << "Scalar constructor test failed!" << std::endl;
    } else {
        std::cout << "Scalar constructor test passed." << std::endl;
    }
}

void test_vector_shape_constructor() {
    std::vector<int> data = {1, 2, 3, 4};
    std::vector<size_t> shape = {2, 2};
    dio::Tensor<int> tensor(data, shape);
    if (tensor.num_dimensions() != 2 || tensor.shape() != shape || tensor.get_data() != data) {
        std::cerr << "Vector and shape constructor test failed!" << std::endl;
    } else {
        std::cout << "Vector and shape constructor test passed." << std::endl;
    }
}

void test_element_access() {
    std::vector<int> data = {1, 2, 3, 4};
    std::vector<size_t> shape = {2, 2};
    dio::Tensor<int> tensor(data, shape);

    if (tensor({0, 0}) != 1 || tensor({0, 1}) != 2 ||
        tensor({1, 0}) != 3 || tensor({1, 1}) != 4) {
        std::cerr << "Element access test failed!" << std::endl;
    } else {
        std::cout << "Element access test passed." << std::endl;
    }
}


void test_element_access_2() {
    std::vector<int> data = {1, 2, 3, 4};
    std::vector<size_t> shape = {2, 2};
    dio::Tensor<int> tensor(data, shape);

    // Access elements
    int val00 = tensor({0, 0}); // Expected 1
    int val01 = tensor({0, 1}); // Expected 2
    int val10 = tensor({1, 0}); // Expected 3
    int val11 = tensor({1, 1}); // Expected 4

    if (val00 != 1 || val01 != 2 || val10 != 3 || val11 != 4) {
        std::cerr << "Element access 2 test failed!" << std::endl;
    } else {
        std::cout << "Element access 2 test passed." << std::endl;
    }
}

void test_addition() {
    std::vector<int> data1 = {1, 2, 3, 4};
    std::vector<size_t> shape = {2, 2};
    dio::Tensor<int> tensor1(data1, shape);

    std::vector<int> data2 = {5, 6, 7, 8};
    dio::Tensor<int> tensor2(data2, shape);

    auto result = tensor1 + tensor2;
    std::vector<int> expected_data = {6, 8, 10, 12};

    if (result.get_data() != expected_data) {
        std::cerr << "Addition test failed!" << std::endl;
    } else {
        std::cout << "Addition test passed." << std::endl;
    }
}

void test_subtraction() {
    std::vector<int> data1 = {5, 6, 7, 8};
    std::vector<size_t> shape = {2, 2};
    dio::Tensor<int> tensor1(data1, shape);

    std::vector<int> data2 = {1, 2, 3, 4};
    dio::Tensor<int> tensor2(data2, shape);

    auto result = tensor1 - tensor2;
    std::vector<int> expected_data = {4, 4, 4, 4};

    if (result.get_data() != expected_data) {
        std::cerr << "Subtraction test failed!" << std::endl;
    } else {
        std::cout << "Subtraction test passed." << std::endl;
    }
}

void test_multiplication() {
    std::vector<int> data1 = {1, 2, 3, 4};
    std::vector<size_t> shape = {2, 2};
    dio::Tensor<int> tensor1(data1, shape);

    std::vector<int> data2 = {2, 3, 4, 5};
    dio::Tensor<int> tensor2(data2, shape);

    auto result = tensor1 * tensor2;
    std::vector<int> expected_data = {2, 6, 12, 20};

    if (result.get_data() != expected_data) {
        std::cerr << "Multiplication test failed!" << std::endl;
    } else {
        std::cout << "Multiplication test passed." << std::endl;
    }
}

void test_division() {
    std::vector<int> data1 = {10, 20, 30, 40};
    std::vector<size_t> shape = {2, 2};
    dio::Tensor<int> tensor1(data1, shape);

    std::vector<int> data2 = {2, 4, 5, 8};
    dio::Tensor<int> tensor2(data2, shape);

    auto result = tensor1 / tensor2;
    std::vector<int> expected_data = {5, 5, 6, 5};

    if (result.get_data() != expected_data) {
        std::cerr << "Division test failed!" << std::endl;
    } else {
        std::cout << "Division test passed." << std::endl;
    }
}

void test_broadcasting_addition() {
    dio::Tensor<int> tensor1(5);  // Scalar tensor
    std::vector<int> data2 = {1, 2, 3, 4};
    std::vector<size_t> shape = {2, 2};
    dio::Tensor<int> tensor2(data2, shape);

    auto result = tensor1 + tensor2;
    std::vector<int> expected_data = {6, 7, 8, 9};

    if (result.get_data() != expected_data) {
        std::cerr << "Broadcasting addition test failed!" << std::endl;
    } else {
        std::cout << "Broadcasting addition test passed." << std::endl;
    }
}

void test_matrix_multiplication() {
    std::vector<int> data1 = {1, 2, 3, 4};  // 2x2 matrix
    std::vector<size_t> shape1 = {2, 2};
    dio::Tensor<int> tensor1(data1, shape1);

    std::vector<int> data2 = {5, 6, 7, 8};  // 2x2 matrix
    std::vector<size_t> shape2 = {2, 2};
    dio::Tensor<int> tensor2(data2, shape2);

    auto result = tensor1.matmul(tensor2);
    std::vector<int> expected_data = {19, 22, 43, 50};  // Result of matrix multiplication

    if (result.get_data() != expected_data) {
        std::cerr << "Matrix multiplication test failed!" << std::endl;
    } else {
        std::cout << "Matrix multiplication test passed." << std::endl;
    }
}

void test_apply_elementwise_function() {
    std::vector<double> data = {0.0, M_PI_2, M_PI, 3*M_PI_2};
    std::vector<size_t> shape = {2, 2};
    dio::Tensor<double> tensor(data, shape);

    auto result = tensor.apply_elementwise_function([](double x) { return std::sin(x); });
    std::vector<double> expected_data = {0.0, 1.0, 0.0, -1.0};

    bool success = true;
    for (size_t i = 0; i < expected_data.size(); ++i) {
        if (std::abs(result.get_data()[i] - expected_data[i]) > 1e-6) {
            success = false;
            break;
        }
    }

    if (!success) {
        std::cerr << "Apply elementwise function test failed!" << std::endl;
    } else {
        std::cout << "Apply elementwise function test passed." << std::endl;
    }
}

void test_invalid_shape() {
    try {
        std::vector<int> data = {1, 2, 3};
        std::vector<size_t> shape = {2, 2};  // Shape does not match data size
        dio::Tensor<int> tensor(data, shape);
        std::cerr << "Invalid shape test failed!" << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cout << "Invalid shape test passed." << std::endl;
    }
}

void test_out_of_bounds_access() {
    try {
        std::vector<int> data = {1, 2, 3, 4};
        std::vector<size_t> shape = {2, 2};
        dio::Tensor<int> tensor(data, shape);
        int value = tensor({2, 0});  // Invalid index
        (void)value;  // Suppress unused variable warning
        std::cerr << "Out-of-bounds access test failed!" << std::endl;
    } catch (const std::out_of_range& e) {
        std::cout << "Out-of-bounds access test passed." << std::endl;
    }
}

int tensor_tests() {
    test_default_constructor();
    test_scalar_constructor();
    test_vector_shape_constructor();
    test_element_access();
    test_element_access_2();
    test_addition();
    test_subtraction();
    test_multiplication();
    test_division();
    test_broadcasting_addition();
    test_matrix_multiplication();
    test_apply_elementwise_function();
    test_invalid_shape();
    test_out_of_bounds_access();

    return 0;
}
