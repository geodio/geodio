from unittest import TestCase

import numpy as np

from core.cell import Linker, Cell
from core.organism.activation_function import SigmoidActivation
from core.organism.node import Node


class TestLinker(TestCase):
    def test_backpropagation(self):
        act_fun = SigmoidActivation()
        node1 = Node(1, 5, 10, act_fun.clone())
        node2 = Node(1, 10, 3, act_fun.clone())

        x = [np.ones((5, 7))]
        dx = np.ones((3, 7))

        # FORWARD
        _ = node2([node1(x)])
        # BACKWARD
        dz = node2.backpropagation(dx)
        backprop_manual = node1.backpropagation(dz)
        gradients_manual = node1.get_gradients() + node2.get_gradients()

        cell1 = Cell(node1, 1, 1)
        cell2 = Cell(node2, 1, 1)
        link = Linker(cell2, cell1)

        # FORWARD
        k = link(x)
        link.get_weights()
        print("OUT Lnk", k.shape)
        # BACKWARD
        backprop_link = link.backpropagation(dx)
        gradients_link = link.get_gradients()

        self.assertEqual(backprop_manual, backprop_link)
        self.assertEqual(gradients_manual, gradients_link)
