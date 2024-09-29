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
#ifndef GEODIO_TEST_SLICE_MEM_H
#define GEODIO_TEST_SLICE_MEM_H

void test_basic_slicing();
void test_implicit_empty_slices();
void test_matrix_multiplication_with_slices();
void test_addition_with_slices();
void test_addition_with_broadcasted_slices();
void test_memory_update();
void test_negative_index_slicing();
void slice_and_update_tests();
void test_broadcasted_matmul();
void test_vector_broadcasting();

#endif //GEODIO_TEST_SLICE_MEM_H
