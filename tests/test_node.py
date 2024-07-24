from unittest import TestCase

import numpy as np

from core.cell.optim.loss import MSEMultivariate
from core.organism.node import Node
from core.organism.activation_function import SigmoidActivation


class TestNode(TestCase):

    def test_single_node(self):
        dim_in = 5
        dim_out = 3
        arity = 1

        activation_function = SigmoidActivation()

        node = Node(arity, dim_in, dim_out, activation_function)

        input_data = [
            [np.array([1, 1, 1, 1, 1])],
            [np.array([0, 1, 1, 0, 1])],
            [np.array([1, 1, 0, 1, 1])],
            [np.array([1, 0, 0, 1, 0])]
        ]
        desired_output = [
            [np.array([1.0, 1.0, 1.0])],
            [np.array([0.0, 1.0, 0.0])],
            [np.array([1.0, 0.0, 1.0])],
            [np.array([0.0, 0.0, 0.0])]
        ]
        loss_function = MSEMultivariate()

        gradient_0 = loss_function.gradient(node, input_data,
                                            desired_output, 0)
        d0 = node.derive(0)
        out0 = d0(input_data[0]).T
        self.assertEqual(out0.shape, (5, 3))
