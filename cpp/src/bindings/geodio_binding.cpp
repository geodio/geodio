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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>
#include "../tensors/AnyTensor.h"
#include "ExecutionEngine.h"

namespace py = pybind11;

PYBIND11_MODULE(tensor_bindings, m) {
    py::class_<dio::AnyTensor>(m, "AnyTensor")
        // Constructors
        .def(py::init<>())  // Default constructor
        .def(py::init<const std::vector<float>&, const std::vector<size_t>&>(), "Create AnyTensor from data and shape")
        .def(py::init<const float&>(), "Create scalar AnyTensor")

        // Copy and move constructors
        .def(py::init<const dio::AnyTensor&>(), "Copy constructor")
//        // Move constructor with return_value_policy::move
//        .def(py::init<dio::AnyTensor&>(), "Move constructor", py::return_value_policy::move)

        // Assignment operators
        .def("__copy__", [](const dio::AnyTensor &self) { return dio::AnyTensor(self); })
        .def("__deepcopy__", [](const dio::AnyTensor &self, py::dict) { return dio::AnyTensor(self); })
        .def("assign", [](dio::AnyTensor &self, const dio::AnyTensor &other) { self = other; }, "Assignment operator")

        // Public methods
        .def("empty", &dio::AnyTensor::empty, "Check if the tensor is empty")
        .def("type", &dio::AnyTensor::type, "Get type info of the stored tensor")
        .def("type_index", &dio::AnyTensor::type_index, "Get the stored type index")
        .def("shape", &dio::AnyTensor::shape, "Get shape of the tensor")

        // Templated get method (uses explicit casting in Python)
        .def("get_float", [](const dio::AnyTensor &self) { return self.get<float>(); })
        .def("get_int", [](const dio::AnyTensor &self) { return self.get<int>(); })
        .def("get_double", [](const dio::AnyTensor &self) { return self.get<double>(); })

        // Templated check type method
        .def("is_float", [](const dio::AnyTensor &self) { return self.is<float>(); })
        .def("is_int", [](const dio::AnyTensor &self) { return self.is<int>(); })
        .def("is_double", [](const dio::AnyTensor &self) { return self.is<double>(); })

        // Element-wise operations
        .def("add", [](dio::AnyTensor &self, const dio::AnyTensor &other) { return self.add(other); })
        .def("subtract", [](dio::AnyTensor &self, const dio::AnyTensor &other) { return self.subtract(other); })
        .def("multiply", [](dio::AnyTensor &self, const dio::AnyTensor &other) { return self.multiply(other); })
        .def("divide", [](dio::AnyTensor &self, const dio::AnyTensor &other) { return self.divide(other); })

        // Scalar operations (use lambdas to avoid `overload_cast`)
        .def("add_scalar", [](dio::AnyTensor &self, float value) { return self.add(value); })
        .def("subtract_scalar", [](dio::AnyTensor &self, float value) { return self.subtract(value); })
        .def("multiply_scalar", [](dio::AnyTensor &self, float value) { return self.multiply(value); })
        .def("divide_scalar", [](dio::AnyTensor &self, float value) { return self.divide(value); })

        // Matrix multiplication
        .def("matmul", &dio::AnyTensor::matmul)

        // Transpose and sum
        .def("transpose", &dio::AnyTensor::transpose, py::arg("axis") = std::vector<size_t>{0})
        .def("sum", &dio::AnyTensor::sum, py::arg("axis") = std::vector<size_t>{0})

        // Slicing
        .def("slice", &dio::AnyTensor::slice)

        // Operators (bind with lambdas)
        .def(py::self + py::self)
        .def(py::self * py::self)
        .def("__sub__", [](const dio::AnyTensor &a, const dio::AnyTensor &b) { return a - b; })
        .def("__truediv__", [](const dio::AnyTensor &a, const dio::AnyTensor &b) { return a / b; })

        // String representation
        .def("__repr__", [](const dio::AnyTensor &self) {
            return "<AnyTensor of type '" + std::string(self.type().name()) + "'>";
        })
        .def("__str__", [](const dio::AnyTensor &self) {
            if (self.is<int>())
                return self.get<int>().to_string();
            if (self.is<float>())
                return self.get<float>().to_string();
            if (self.is<double>())
                return self.get<double>().to_string();
            return std::string("Empty tensor");
        });

    py::enum_<dio::OperandType>(m, "OperandType")
        .value("Constant", dio::OperandType::Constant)
        .value("Variable", dio::OperandType::Variable)
        .value("Weight", dio::OperandType::Weight)
        .value("Seq", dio::OperandType::Seq)
        .value("Tick", dio::OperandType::Tick)
        .value("getTime", dio::OperandType::getTime)
        .value("Jump", dio::OperandType::Jump)
        .value("Label", dio::OperandType::Label)
        .value("Condition", dio::OperandType::Condition)
        .value("LessThan", dio::OperandType::LessThan)
        .value("GreaterThan", dio::OperandType::GreaterThan)
        .value("Equal", dio::OperandType::Equal)
        .value("LessThanOrEqual", dio::OperandType::LessThanOrEqual)
        .value("GreaterThanOrEqual", dio::OperandType::GreaterThanOrEqual)
        .value("Add", dio::OperandType::Add)
        .value("Multiply", dio::OperandType::Multiply)
        .value("Sigmoid", dio::OperandType::Sigmoid)
        .value("LinearTransformation", dio::OperandType::LinearTransformation)
        .value("Identity", dio::OperandType::Identity);

    // Binding for Operand class
    py::class_<dio::Operand>(m, "Operand")
        .def(py::init<>())  // Default constructor
        .def(py::init<dio::OperandType, int, const std::vector<int>&>(), "Constructor with args", py::arg("op_type"), py::arg("id"), py::arg("inputs"))
        .def_readwrite("op_type", &dio::Operand::op_type)
        .def_readwrite("id", &dio::Operand::id)
        .def_readwrite("inputs", &dio::Operand::inputs);

    // Binding for ComputationalGraph class
    py::class_<dio::ComputationalGraph>(m, "ComputationalGraph")
        .def(py::init<int>(), py::arg("rootId") = 0)
        .def_readwrite("root_id", &dio::ComputationalGraph::root_id)
        .def_readwrite("operands", &dio::ComputationalGraph::operands)
        .def_readwrite("weights", &dio::ComputationalGraph::weights)
        .def_readwrite("gradients", &dio::ComputationalGraph::gradients)
        .def_readwrite("constants", &dio::ComputationalGraph::constants)
        .def_readwrite("var_map", &dio::ComputationalGraph::var_map);

    // Binding for ExecutionEngine class
    py::class_<dio::ExecutionEngine>(m, "ExecutionEngine")
        .def_static("forward", &dio::ExecutionEngine::forward, py::arg("graph"), py::arg("output_operand_id"), py::arg("args") = std::vector<dio::a_tens>{})
        .def_static("backward", &dio::ExecutionEngine::backward, py::arg("graph"), py::arg("output_operand_id"), py::arg("loss_gradient"), py::arg("args") = std::vector<dio::a_tens>{})
        .def_static("optimize", &dio::ExecutionEngine::optimize, py::arg("graph"), py::arg("input_data"), py::arg("target_data"), py::arg("args"))
        .def_static("evaluate_condition", &dio::ExecutionEngine::evaluate_condition, py::arg("graph"), py::arg("operand"), py::arg("inputs"));
}
