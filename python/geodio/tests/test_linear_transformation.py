from unittest import TestCase

import numpy as np

from geodio.core import LinearTransformation, b_var


class TestLinearTransformation(TestCase):

    def test_forward(self):
        dim_in = 5
        dim_out = 3
        linear_transformation = LinearTransformation(dim_in, dim_out, [b_var()])
        input_data = [np.array([1, 1, 1, 1, 1])]
        output = linear_transformation(input_data)
        self.assertEqual(output.shape, (dim_out,))

    # def test_derivative_w(self):
    #     dim_in = 5
    #     dim_out = 3
    #     linear_transformation = LinearTransformation(dim_in, dim_out, [b_var()])
    #     input_data = [np.array([1, 1, 1, 1, 1])]
    #     derivative_w = linear_transformation.derive(0, by_weights=True)
    #     expected_shape = (dim_in, )
    #     self.assertEqual(derivative_w(input_data).shape, expected_shape)
    #
    # def test_derivative_x(self):
    #     dim_in = 5
    #     dim_out = 3
    #     linear_transformation = LinearTransformation(dim_in, dim_out, [b_var()])
    #     input_data = [np.array([1, 1, 1, 1, 1])]
    #     derivative_x = linear_transformation.derive(0, by_weights=False)
    #     expected_shape = (dim_out, dim_in)
    #     self.assertEqual(derivative_x(input_data).shape, expected_shape)
    #
    # def test_derivative_b(self):
    #     dim_in = 5
    #     dim_out = 3
    #     linear_transformation = LinearTransformation(dim_in, dim_out, [b_var()])
    #     input_data = [np.array([1, 1, 1, 1, 1])]
    #     derivative_b = linear_transformation.derive(1, by_weights=True)
    #     expected_shape = (dim_out, dim_out)
    #     self.assertEqual(derivative_b(input_data).shape, expected_shape)
    #
    # def test_value_1(self):
    #     dim_in = 1
    #     dim_out = 1
    #     w = np.array([[313]])
    #     lt = LinearTransformation(dim_in, dim_out, [b_var()])
    #     input_data = [np.array([[2]])]
    #     lt.weight.set(w)
    #     derivative = lt.derive(0, False)
    #     dx_r = derivative(input_data)
    #     self.assertEqual(dx_r, w)
    #
    #     derivative = lt.derive(0, True)
    #     dx_r = derivative(input_data)
    #     self.assertEqual(dx_r, input_data)
    #
    #     derivative = lt.derive(1, True)
    #     dx_r = derivative(input_data)
    #     self.assertEqual(dx_r, np.array([0]))
    #
    # def test_value_2(self):
    #     dim_in = 3
    #     dim_out = 3
    #     w = np.array([
    #         [-1, -1, -1],
    #         [1, 0, 1],
    #         [-1, -1, -1]
    #     ])
    #     lt = LinearTransformation(dim_in, dim_out, [b_var()])
    #     input_data = [np.array([2, 2, 2])]
    #     lt.weight.set(w)
    #     derivative = lt.derive(0, False)
    #     dx_r = derivative(input_data)
    #     self.assertTrue(np.array_equal(dx_r, w), f"{dx_r} != {w}")
    #
    #     derivative = lt.derive(0, True)
    #     dx_r = derivative(input_data)
    #     self.assertTrue(np.array_equal(dx_r, input_data[0]),
    #                     f"{dx_r} != {input_data[0]}")
    #
    #     derivative = lt.derive(1, True)
    #     dx_r = derivative(input_data)
    #     self.assertTrue(np.array_equal(dx_r[0], np.array([0, 0, 0])),
    #                     f"{dx_r[0]} != [0, 0, 0]")
