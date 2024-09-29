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
#include <cassert>
#include "../src/tensors/Tensor.h"

void test_basic_slicing() {
    // Create a 2x3 tensor
    dio::Tensor<float> tensor({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    }, {2, 3});

    // Slice the tensor: select first row (slice(0, 1)) and all columns
    dio::Tensor<float> slice = tensor.slice({dio::Slice(0, 1), dio::Slice(0, 3)});

    // Expected output is [1.0, 2.0, 3.0]
    assert(slice({0, 0}) == 1.0f);
    assert(slice({0, 1}) == 2.0f);
    assert(slice({0, 2}) == 3.0f);

    std::cout << "Basic slicing test passed!" << std::endl;
}
void test_implicit_empty_slices() {
    dio::Tensor<float> tensor({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    }, {2, 3});

    // Accessing the entire first row implicitly
    dio::Tensor<float> slice = tensor.slice({dio::Slice(0, 1)});  // Should access first row entirely
    assert(slice({0, 0}) == 1.0f);
    assert(slice({0, 1}) == 2.0f);
    assert(slice({0, 2}) == 3.0f);

    std::cout << "Implicit empty slices test passed!" << std::endl;
}

void test_matrix_multiplication_with_slices() {
    dio::Tensor<float> mat1({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    }, {3, 3});

    dio::Tensor<float> mat2({
        9.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 4.0f,
        3.0f, 2.0f, 1.0f
    }, {3, 3});

    dio::Tensor<float> slice1 = mat1.slice({dio::Slice(1, 3), dio::Slice(1, 3)});
    dio::Tensor<float> slice2 = mat2.slice({dio::Slice(1, 3), dio::Slice(1, 3)});
    //    std::cout << slice1 << std::endl;
    //    std::cout << slice2 << std::endl;
    dio::Tensor<float> result = slice1.matmul(slice2);
    std::cout << result << std::endl;
    // Expected result: [[37, 26], [58, 41]]
    assert(result({0, 0}) == 37.0f);
    assert(result({0, 1}) == 26.0f);
    assert(result({1, 0}) == 58.0f);
    assert(result({1, 1}) == 41.0f);

    std::cout << "Matrix multiplication with slices test passed!" << std::endl;
}

void test_addition_with_slices() {
    dio::Tensor<float> mat1({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    }, {3, 3});

    dio::Tensor<float> mat2({
        9.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 4.0f,
        3.0f, 2.0f, 1.0f
    }, {3, 3});

    dio::Tensor<float> slice1 = mat1.slice({dio::Slice(1, 3), dio::Slice(1, 3)});
    dio::Tensor<float> slice2 = mat2.slice({dio::Slice(1, 3), dio::Slice(1, 3)});
    //    std::cout << slice1 << std::endl;
    //    std::cout << slice2 << std::endl;
    dio::Tensor<float> result = slice1.add(slice2);
    // std::cout << result << std::endl;
    // Expected result: [[10, 10], [10, 10]]
    assert(result({0, 0}) == 10.0f);
    assert(result({0, 1}) == 10.0f);
    assert(result({1, 0}) == 10.0f);
    assert(result({1, 1}) == 10.0f);

    std::cout << "Addition with slices test passed!" << std::endl;
}

void test_addition_with_broadcasted_slices() {
    dio::Tensor<float> mat1({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f,
        7.0f, 8.0f, 9.0f
    }, {3, 3});

    dio::Tensor<float> mat2({
        9.0f, 8.0f, 7.0f,
        6.0f, 5.0f, 4.0f,
        3.0f, 2.0f, 1.0f
    }, {3, 3});

    dio::Tensor<float> slice1 = mat1.slice({dio::Slice(1, 3), dio::Slice(1, 3)});
    dio::Tensor<float> slice2 = mat2.slice({dio::Slice(1, 2), dio::Slice(1, 2)});
    //    std::cout << slice1 << std::endl;
    //    std::cout << slice2 << std::endl;
    dio::Tensor<float> result = slice1.add(slice2);
    // std::cout << result << std::endl;
    // Expected result: [[10, 11], [13, 14]]
    assert(result({0, 0}) == 10.0f);
    assert(result({0, 1}) == 11.0f);
    assert(result({1, 0}) == 13.0f);
    assert(result({1, 1}) == 14.0f);

    std::cout << "Addition with broadcasting slices test passed!" << std::endl;
}

void test_negative_index_slicing() {
    // Create a 4x5 tensor
    dio::Tensor<int> tensor({
        1, 2, 3, 4, 5,
        6, 7, 8, 9, 10,
        11, 12, 13, 14, 15,
        16, 17, 18, 19, 20
    }, {4, 5});

    // Slice with negative indices (select last 2 rows and last 3 columns)
    dio::Tensor<int> slice = tensor.slice({dio::Slice(-2, 4), dio::Slice(-3, 5)});

    // Expected output is [[13, 14, 15], [18, 19, 20]]
    std::cout << "Sliced output:" << std::endl;
    std::cout << slice << std::endl;  // You can implement a print method for better formatting
    assert(slice({0, 0}) == 13);
    assert(slice({0, 1}) == 14);
    assert(slice({0, 2}) == 15);
    assert(slice({1, 0}) == 18);
    assert(slice({1, 1}) == 19);
    assert(slice({1, 2}) == 20);

    std::cout << "Negative index slicing test passed!" << std::endl;
}

void test_memory_update() {
    // Create a 2x3 tensor
    dio::Tensor<float> tensor({
        1.0f, 2.0f, 3.0f,
        4.0f, 5.0f, 6.0f
    }, {2, 3});

    // Create a slice (view)
    dio::Tensor<float> slice = tensor.slice({dio::Slice(0, 2), dio::Slice(0, 2)}); // View on first 2 columns

    // Update the slice
    slice({0, 0}) = 10.0f;

    // Check if the original tensor is updated
    assert(tensor({0, 0}) == 10.0f);

    std::cout << "Memory update test passed!" << std::endl;
}

void test_broadcasted_matmul() {
    dio::Tensor<float> mat({
        1.0f, 2.0f,
        3.0f, 4.0f
    }, {2, 2});

    dio::Tensor<float> vec({5.0f, 6.0f}, {2});  // Vector

    dio::Tensor<float> result = mat.matmul(vec);
    std::cout << result << std::endl;

    // Expected result: [17, 39] (matrix-vector multiplication)
    assert(result({0, 0}) == 17.0f);
    assert(result({1, 0}) == 39.0f);

    std::cout << "Broadcasted matmul test passed!" << std::endl;
}

void test_vector_broadcasting() {
    dio::Tensor<float> vec1({1.0f, 2.0f, 3.0f}, {3});
    dio::Tensor<float> vec2({4.0f}, {1});

    dio::Tensor<float> result = vec1.matmul(vec2);
    std::cout << result << std::endl;
    // Expected result: [[24]] (broadcasted scalar multiplication)
    assert(result({0, 0}) == 24.0f);

    std::cout << "Vector broadcasting test passed!" << std::endl;
}

int slice_and_update_tests() {
    test_basic_slicing();
    test_implicit_empty_slices();
    test_negative_index_slicing();
    test_matrix_multiplication_with_slices();
    test_addition_with_slices();
    test_addition_with_broadcasted_slices();
    test_memory_update();
    test_broadcasted_matmul();
    test_vector_broadcasting();
    std::cout << "All tests passed!" << std::endl;
    return 0;
}