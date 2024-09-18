#include "test_tensors.h"
#include "test_tensor_db.h"     // Include the database tests

void run_tensor() {
    tensor_tests();
}

void run_tensor_db() {
    tensor_db_tests();
}

void run_all() {
    run_tensor();
    run_tensor_db();
}