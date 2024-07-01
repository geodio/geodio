from unittest import TestCase

from src.cell.cell import Cell
from src.cell.default_functors import Power, Add, Prod
from src.cell.operands.constant import Constant
from src.cell.operands.variable import Variable


class TestCell(TestCase):

    def test_derivative_power(self):
        children = [Variable(0), Constant(2)]
        root = Power([])
        root.children = children
        cell = Cell(root, 1, 2)
        derivative_cell = cell.derive(0, False)
        self.assertEqual(cell([7]), 49)
        self.assertEqual(derivative_cell([7]), 14)

        self.assertEqual(cell([-10]), 100)
        self.assertEqual(derivative_cell([-10]), -20)

    def test_derivative_sum(self):
        children = [Variable(0), Constant(2), Variable(0)]
        root = Add(children, 3)
        cell = Cell(root, 1, 2)
        derivative_cell = cell.derive(0, False)
        self.assertEqual(cell([7]), 16)
        self.assertEqual(derivative_cell([7]), 2)

        self.assertEqual(cell([-10]), -18)
        self.assertEqual(derivative_cell([-10]), 2)

    def test_derivative_prod(self):
        children = [Variable(0), Variable(0)]
        root = Prod(children)
        cell = Cell(root, 1, 2)
        derivative_cell = cell.derive(0, False)
        self.assertEqual(49, cell([7]))
        self.assertEqual(14, derivative_cell([7]))

        self.assertEqual(100, cell([-10]))
        self.assertEqual(-20, derivative_cell([-10]))


