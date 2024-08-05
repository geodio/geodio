from unittest import TestCase

from core.cell.cell import Cell
from core.cell.operands.collections import Prod, Add, Power
from core.cell.operands.constant import Constant
from core.cell.operands.variable import Variable
from core.cell.operands.weight import Weight
from core.organism.router import Router


class TestRouter(TestCase):

    def test_derivative_router(self):
        roots = [
            Prod([Variable(0), Constant(7)]),
            Add([Variable(0), Constant(2)], 2),
            Power([Variable(0), Constant(3)]),
        ]
        cells = [Cell(root, 1, 1) for root in roots]
        var_input = [5]
        outputs = [cell.state_update(var_input) for cell in cells]
        output_1, output_2, output_3 = 35, 7, 125
        self.assertEqual(outputs[0], output_1)
        self.assertEqual(outputs[1], output_2)
        self.assertEqual(outputs[2], output_3)

        w1, w2, w3 = 0.2, 1, 0.4
        weights = {0: Weight(w1), 1: Weight(w2), 2: Weight(w3)}
        router = Router(cells, weights)
        router_output = router([])
        desired_output = w1 * output_1 + w2 * output_2 + w3 * output_3
        self.assertEqual(desired_output, router_output)

        router.get_weights()

        r_derive_s_0 = router.derive(0)
        desired_output_s_0 = w1 * 1.0 + w2 * output_2 + w3 * output_3
        router_output_s_0 = r_derive_s_0([])
        self.assertEqual(desired_output_s_0, router_output_s_0)

        r_derive_w_2 = router.derive(4)
        desired_output_w_2 = w1 * output_1 + 1.0 * output_2 + w3 * output_3
        router_output_w_2 = r_derive_w_2([])
        self.assertEqual(desired_output_w_2, router_output_w_2)
