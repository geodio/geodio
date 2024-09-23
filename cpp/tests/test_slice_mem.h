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
