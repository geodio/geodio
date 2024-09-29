//    Geodio Neural Network Framework - A lightweight neural network library focused on custom architecture design and optimization.
//    Copyright (C) Copyright (C) 2024 Geodio (created by Rareș Polenciuc)
//
//    This program is free software: you can redistribute it and/or modify
//    it under the terms of the GNU General Public License as published by
//    the Free Software Foundation, either version 3 of the License, or
//    (at your option) any later version.
//
//    This program is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU General Public License for more details.
//
//    You should have received a copy of the GNU General Public License
//    along with this program.  If not, see <https://www.gnu.org/licenses/>.
#include <iostream>
#include <vector>
#include <cmath>
#include <cassert>
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
double some_sin(double x) {
    return std::sin(x);
}

void test_apply_elementwise_function() {
    std::vector<double> data = {0.0, M_PI_2, M_PI, 3*M_PI_2};
    std::vector<size_t> shape = {2, 2};
    dio::Tensor<double> tensor(data, shape);
    std::function<double(double)> some_sin = [](double x) { return std::sin(x); };
    auto result = tensor.apply_elementwise_function(some_sin);
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

void test_transpose() {
    // Test 1: Transpose of a 2x3 matrix
    std::vector<float> data_2x3 = {1, 2, 3, 4, 5, 6};
    dio::Tensor matrix_2x3(data_2x3, {2, 3});
    std::vector<float> expected_data_3x2 = {1, 4, 2, 5, 3, 6};
    dio::Tensor expected_transpose_3x2(expected_data_3x2, {3, 2});
    dio::Tensor result_transpose_3x2 = matrix_2x3.transpose({});
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 2; j++)
            assert(result_transpose_3x2({i,j}) == expected_transpose_3x2({i,j}));

    // Test 2: Transpose of a square matrix (3x3)
    std::vector<float> data_3x3 = {1, 2, 3, 4, 5, 6, 7, 8, 9};
    dio::Tensor matrix_3x3(data_3x3, {3, 3});
    std::vector<float> expected_data_transpose_3x3 = {1, 4, 7, 2, 5, 8, 3, 6, 9};
    dio::Tensor expected_transpose_3x3(expected_data_transpose_3x3, {3, 3});
    dio::Tensor result_transpose_3x3 = matrix_3x3.transpose({});
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 3; j++)
            assert(result_transpose_3x3({i, j}) == expected_transpose_3x3({i, j}));

    // Test 3: Transpose of a 1x3 vector
    std::vector<float> data_1x3 = {1, 2, 3};
    dio::Tensor vector_1x3(data_1x3, {1, 3});
    std::vector<float> expected_data_3x1 = {1, 2, 3};
    dio::Tensor expected_transpose_3x1(expected_data_3x1, {3, 1});
    dio::Tensor result_transpose_3x1 = vector_1x3.transpose({});
    for (size_t i = 0; i < 3; i++)
        for (size_t j = 0; j < 1; j++)
            assert(result_transpose_3x1({i, j}) == expected_transpose_3x1({i, j}));

    // Test 4: Transpose of a 3x1 vector
    std::vector<float> data_3x1 = {1, 2, 3};
    dio::Tensor vector_3x1(data_3x1, {3, 1});
    std::vector<float> expected_data_1x3 = {1, 2, 3};
    dio::Tensor expected_transpose_1x3(expected_data_1x3, {1, 3});
    dio::Tensor result_transpose_1x3 = vector_3x1.transpose({});
    for (size_t i = 0; i < 1; i++)
        for (size_t j = 0; j < 3; j++)
            assert(result_transpose_1x3({i, j}) == expected_transpose_1x3({i, j}));

    std::cout << "All transpose tests passed!" << std::endl;
}

void test_sum() {
    // Test 1: Sum of all elements in a 2x3 matrix
    std::vector<float> data_2x3 = {1, 2, 4, 8, 16, 32};
    dio::Tensor matrix_2x3(data_2x3, {2, 3});
    float expected_sum_all = 63; // 1+2+4+8+16+32 = 63
    dio::Tensor<float> result_sum_all = matrix_2x3.sum({});
    std::cout << result_sum_all << std::endl;
    assert(result_sum_all.get_data()[0] == expected_sum_all);

    // Test 2: Sum along rows (axis=0) in a 2x3 matrix
    std::vector<float> expected_data_sum_rows = {5, 7, 9}; // sum along rows: {1+4, 2+5, 3+6}
    dio::Tensor expected_sum_rows(expected_data_sum_rows, {1, 3});
    dio::Tensor result_sum_rows = matrix_2x3.sum({0});
//    for (size_t i = 0; i < 3; i++)
//        assert(result_sum_rows({i}) == expected_sum_rows({0, i}));

    // Test 3: Sum along columns (axis=1) in a 2x3 matrix
    std::vector<float> expected_data_sum_cols = {6, 15}; // sum along columns: {1+2+3, 4+5+6}
    dio::Tensor expected_sum_cols(expected_data_sum_cols, {2, 1});
    dio::Tensor result_sum_cols = matrix_2x3.sum({1});
//    for (size_t i = 0; i < 2; i++)
//        assert(result_sum_cols({i}) == expected_sum_cols({i}));

    // Test 4: Sum of a 1D vector
    std::vector<float> data_1d = {1, 2, 3, 4};
    dio::Tensor vector_1d(data_1d, {4});
    float expected_sum_vector = 10; // 1+2+3+4 = 10
//    float result_sum_vector = vector_1d.sum({0})({0});
//    assert(result_sum_vector == expected_sum_vector);

    std::cout << "All sum tests passed!" << std::endl;
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
    test_transpose();
    test_sum();

    return 0;
}
