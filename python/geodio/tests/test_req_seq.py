from unittest import TestCase

import numpy as np

from geodio.core import EmptyCell, Variable, RecSeq, Constant, Stateful
from geodio.core.cell.operands.collections.builtins.conditionals.if_cond import \
    if_cond


class TestRecSeq(TestCase):

    def test_summation(self):
        cell = EmptyCell(1)
        cell.set_root(cell.get_state_weight() + Variable(0))

        rec_seq = RecSeq(cell)

        args = np.random.random(10)
        expected = sum(args)
        actual = rec_seq(args)
        self.assertEqual(expected, actual)

    def test_cond_prod(self):
        cell = EmptyCell(1)
        state = cell.get_state_weight()
        v_0 = Variable(0)
        c_0 = Constant(0)

        cell.set_root(
            if_cond(state.equals(c_0), v_0, state * v_0)
        )

        rec_seq = RecSeq(cell)

        # adding a small epsilon to assure non zero values
        args = np.random.random(10) + 1e-10
        expected = 1
        for arg in args:
            expected *= arg
        actual = rec_seq(args)
        self.assertEqual(expected, actual)

    def test_summation_states(self):
        cell = EmptyCell(1)
        cell.set_root(cell.get_state_weight() + Variable(0))

        rec_seq = RecSeq(cell)

        args = [1, 2, 3, 4]
        expected = sum(args)
        actual = rec_seq(args)
        self.assertEqual(expected, actual)
        acc = 0

        for i, op in enumerate(rec_seq):
            print(i)
            acc += args[i]
            self.assertTrue(isinstance(op, Stateful))
            op: Stateful = op
            self.assertEqual(op.state, acc)
