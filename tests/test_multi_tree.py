from unittest import TestCase

from core.cell.cell import Cell
from core.cell.operands.collections import Power, Add, Prod
from core.cell.operands.constant import Constant
from core.cell.operands.variable import Variable
from core.cell.train.forest import Forest


class TestMultiTree(TestCase):
    def test_multi_tree(self):
        var_x = 2
        power_with = 7
        add_with = 3
        prod_with = 4
        chain1 = Cell(Power([Variable(0), Constant(power_with)]), 1, 2)
        chain2 = Cell(Add([Variable(0), Constant(add_with)], 2), 1, 2)
        chain3 = Cell(Prod([Variable(0), Constant(prod_with)]), 1, 2)
        mtree = Forest([chain1, chain2, chain3], 1)
        expected = [var_x ** power_with, var_x + add_with, var_x * prod_with]
        actual = mtree(var_x)
        self.assertEqual(expected, actual)
