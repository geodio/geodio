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