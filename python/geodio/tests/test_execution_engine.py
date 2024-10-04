from unittest import TestCase

import numpy as np

from geodio.core import (LinearTransformation, b_var, GraphWrapper, Tensor,
                         const, Add)
import geodio.geodio_bindings as geodio_bindings


class TestExecutionEngine(TestCase):

    def test_forward(self):
        dim_in = 1
        dim_out = 1
        input_data = np.array([[1.0]])
        constant = const(input_data)
        linear_transformation = LinearTransformation(dim_in, dim_out,
                                                     [constant])
        geodio_bindings.initialize_operations()
        # in_tens = Tensor(input_data)
        graph = GraphWrapper()
        t_r_py = Tensor(linear_transformation([input_data]))
        linear_transformation.subscribe_to_graph(graph)

        result = geodio_bindings.ExecutionEngine.forward(graph.graph, 0,
                                                         [])
        self.assertEqual(str(t_r_py), str(result))

    def test_forward_bvar(self):
        dim_in = 2
        dim_out = 2
        input_data = np.array([[1.0, 0.0], [3.0, 2.0]])
        linear_transformation = LinearTransformation(dim_in, dim_out,
                                                     [b_var()])
        geodio_bindings.initialize_operations()
        in_tens = Tensor(input_data)
        graph = GraphWrapper()
        t_r_py = Tensor(linear_transformation([input_data]))
        linear_transformation.subscribe_to_graph(graph)

        result = geodio_bindings.ExecutionEngine.forward(graph.graph, 0,
                                                         [in_tens.tensor])
        self.assertEqual(str(t_r_py), str(result))

    def test_add(self):
        input_data = np.array([1.0, 2.0])
        constant = const(np.array([5.0, 4.0]))
        addition = Add([constant, b_var()], 2)
        geodio_bindings.initialize_operations()
        in_tens = Tensor(input_data)
        graph = GraphWrapper()
        t_r_py = Tensor(addition([input_data]))
        addition.subscribe_to_graph(graph)

        result = geodio_bindings.ExecutionEngine.forward(graph.graph, 0,
                                                         [in_tens.tensor])

        self.assertEqual(str(t_r_py), str(result))

    def test_constant(self):

        constant = const(69.0)
        graph = GraphWrapper()
        result_python = constant([])
        constant.subscribe_to_graph(graph)
        result_cpp = geodio_bindings.ExecutionEngine.forward(graph.graph, 0,
                                                             [])
        tensor_result_python = Tensor(result_python)
        self.assertEqual(str(tensor_result_python), str(result_cpp))

        constant = const([69.0, 213123.0])
        graph = GraphWrapper()
        result_python = constant([])
        constant.subscribe_to_graph(graph)
        result_cpp = geodio_bindings.ExecutionEngine.forward(graph.graph, 0,
                                                             [])
        tensor_result_python = Tensor(result_python)
        self.assertEqual(str(tensor_result_python), str(result_cpp))
