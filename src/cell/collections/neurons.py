import numpy as np

from src.cell.cell import Cell
from src.cell.collections.bank import CellBank
from src.cell.collections.builtin_functors import Div, Add, Power, Prod, Dot
from src.cell.collections.functors import CollectionBasedFunctors
from src.cell.operands.constant import ONE, E, MINUS_ONE
from src.cell.operands.variable import Variable
from src.cell.operands.weight import Weight


class Neuron(Cell):
    pass


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class Sigmoid(Neuron):
    def __init__(self, input_style):
        self.W = Weight(np.zeros_like(input_style))
        self.B = Weight(0)
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


NEURONS = CellBank()
NEURONS.add_cell(Sigmoid(0.0))
