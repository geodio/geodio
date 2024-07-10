import random

import numpy as np

from core.cell.cell import Cell
from core.cell.collections.bank import CellBank
from core.cell.collections.builtin_functors import Div, Add, Power, Prod, Dot, \
    BuiltinFunctor
from core.cell.collections.functors import CollectionBasedFunctors, Functor
from core.cell.operands.constant import ONE, E, MINUS_ONE
from core.cell.operands.variable import Variable
from core.cell.operands.weight import Weight


class Neuron(Cell):
    pass


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Sigmoid(Neuron):
    def __init__(self, input_style):
        self.W = Weight(np.zeros_like(input_style), adaptive_shape=True)
        self.B = Weight(random.random())
        self.input = Add([
            Dot([
                self.W,
                Variable(0)
            ]),
            self.B
        ], 2)
        self.activation_function = Div(
            [
                ONE,
                Add(
                    [
                        ONE,
                        Power([
                            E,
                            Prod([
                                MINUS_ONE,
                                self.input
                            ])
                        ])
                    ],
                    2
                )
            ]
        )
        super().__init__(self.activation_function, 1, 7)
        self.frozen = "SIGMOID"
        self.set_optimization_risk(True)

    def clone(self) -> 'Sigmoid':
        clone_sigmoid = Sigmoid(self.W.get())
        clone_sigmoid.B.set(random.random())
        return clone_sigmoid


NEURONS = CellBank()
NEURONS.add_cell(Sigmoid(0.0))
