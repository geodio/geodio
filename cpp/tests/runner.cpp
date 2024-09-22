#include "test_tensors.h"
#include "test_tensor_db.h"     // Include the database tests
#include "test_execution_engine.h"
#include "test_slice_mem.h"

void run_tensor() {
    tensor_tests();
}

void run_tensor_db() {
    tensor_db_tests();
}

void run_execution_engine() {
    execution_engine_tests();
}

void run_slicing() {
    slice_and_update_tests();
}


void run_all() {
    run_tensor();
    run_tensor_db();
    run_execution_engine();
    run_slicing();
}