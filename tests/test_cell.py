import sys
from unittest import TestCase

from core.cell.cell import Cell
from core.cell import Power, Add, Prod
from core.cell.train.loss import MSE
from core.cell.operands.constant import Constant
from core.cell.operands.variable import Variable
from core.cell.operands.weight import Weight


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

    def test_simple_optimization(self):
        w = Weight(1.0)
        children = [w, Constant(2)]
        root = Prod(children)
        cell = Cell(root, 0, 1)
        derivative_cell = cell.derive(0, True)
        self.assertEqual(2.0, cell([7]))
        self.assertEqual(2.0, derivative_cell([7]))

        self.assertEqual(2.0, cell([-10]))
        self.assertEqual(2.0, derivative_cell([-10]))

        mse = MSE()
        X = [[], []]
        Y = [80, 80]
        gradient = mse.gradient(cell, X, Y, 0)
        self.assertAlmostEqual(- 312.0, gradient)
        mse.evaluate(cell, X, Y)
        self.assertNotEquals(0.0, cell.error)
        cell.set_optimization_risk(True)
        cell.optimize_values(mse, X, Y, max_iterations=50,
                             min_error=sys.maxsize)
        gradient = mse.gradient(cell, X, Y, 0)
        self.assertAlmostEqual(40.0, w.get())
        self.assertAlmostEqual(0.0, gradient)
        self.assertAlmostEqual(0.0, cell.error)

    def test_medium_optimization(self):
        w = Weight(7.0)
        children = [Variable(0), w]
        root = Power(children)
        cell = Cell(root, 1, 1)
        derivative_cell = cell.derive(0, False)
        self.assertEqual(3 ** 7, cell([3]))
        print(derivative_cell)
        self.assertEqual(7 * (2 ** 6), derivative_cell([2]))
        # derivative_cell = cell.derive(0, True)
        mse = MSE()
        X = [[2], [3]]
        Y = [2 ** 6.9, 3 ** 6.9]
        gradient = mse.gradient(cell, X, Y, 0)
        self.assertNotEqual(0, gradient)
        mse.evaluate(cell, X, Y)
        self.assertNotEqual(0.0, cell.error)

        cell.optimize_values(mse, X, Y, max_iterations=1000,
                             min_error=sys.maxsize)
        print(cell)
        gradient = mse.gradient(cell, X, Y, 0)
        self.assertTrue(abs(6.9 - w.get()) < 1e-5)
        # self.assertAlmostEqual(0.0, gradient)
        self.assertTrue(cell.error < 1e-5)
        print(w.get())


"""
add_2(
    prod(
        prod(
            7.0, power(x[0], add_2(7.0, -1))
        ), 
        sub(x[0], x[0])
    ), 
    prod(
        prod(power(x[0], 7.0), log(x[0])), 1
    )
) 
"""
