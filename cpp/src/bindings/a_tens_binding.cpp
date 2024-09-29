#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "AnyTensor.h"  // Include your header files

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
        .def("__add__", [](const dio::AnyTensor &a, const dio::AnyTensor &b) { return a + b; })
        .def("__sub__", [](const dio::AnyTensor &a, const dio::AnyTensor &b) { return a - b; })
        .def("__mul__", [](const dio::AnyTensor &a, const dio::AnyTensor &b) { return a * b; })
        .def("__truediv__", [](const dio::AnyTensor &a, const dio::AnyTensor &b) { return a / b; })

        // String representation
        .def("__repr__", [](const dio::AnyTensor &self) {
            return "<AnyTensor of type '" + std::string(self.type().name()) + "'>";
        });
}
