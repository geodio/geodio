from unittest import TestCase

import numpy as np

from core.organism.node import LinearTransformation


class TestLinearTransformation(TestCase):

    def test_forward(self):
        dim_in = 5
        dim_out = 3
        linear_transformation = LinearTransformation(dim_in, dim_out)
        input_data = [np.array([1, 1, 1, 1, 1])]
        output = linear_transformation(input_data)
        self.assertEqual(output.shape, (dim_out,))

    def test_derivative_w(self):
        dim_in = 5
        dim_out = 3
        linear_transformation = LinearTransformation(dim_in, dim_out)
        input_data = [np.array([1, 1, 1, 1, 1])]
        derivative_w = linear_transformation.derive(0, by_weights=True)
        expected_shape = (dim_out, dim_in)
        self.assertEqual(derivative_w(input_data).shape, expected_shape)

    def test_derivative_x(self):
        dim_in = 5
        dim_out = 3
        linear_transformation = LinearTransformation(dim_in, dim_out)
        input_data = [np.array([1, 1, 1, 1, 1])]
        derivative_x = linear_transformation.derive(0, by_weights=False)
        expected_shape = (dim_out, dim_in)
        self.assertEqual(derivative_x(input_data).shape, expected_shape)

    def test_derivative_b(self):
        dim_in = 5
        dim_out = 3
        linear_transformation = LinearTransformation(dim_in, dim_out)
        input_data = [np.array([1, 1, 1, 1, 1])]
        derivative_b = linear_transformation.derive(1, by_weights=True)
        expected_shape = (dim_out,)
        self.assertEqual(derivative_b(input_data).shape, expected_shape)
