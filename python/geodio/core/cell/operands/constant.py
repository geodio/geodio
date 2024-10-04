from typing import Any

import numpy as np

from geodio.core.cell.operands.operand import Operand
from geodio.core.cpp_wrappers.tensor_wrapper import Tensor
import geodio


class Constant(Operand):
    def __invert__(self):
        return self

    def clone(self) -> "Operand":
        return self

    def to_python(self) -> str:
        return f"{self.__value}"

    def __init__(self, weight):
        """

        :param weight: the value stored by this constant.
        """
        super().__init__(arity=0)
        self.__value = weight

    def __call__(self, args, meta_args=None):
        return self.__value

    def d(self, dx):
        return ZERO

    def d_w(self, dw):
        return ZERO

    @staticmethod
    def from_weight(w):
        return Constant(w.get())

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def __eq__(self, other):
        if isinstance(other, Constant):
            return self.__value == other.__value

    def subscribe_to_graph(self, graph_wrapper, operand_type=None):
        super().subscribe_to_graph(graph_wrapper, self.get_operand_type())
        graph_wrapper.add_constant(self.graph_id, Tensor(self.__value))
        return self.graph_id

    def get_operand_type(self):
        return geodio.geodio_bindings.OperandType.Constant


ZERO = Constant(0.0)
ONE = Constant(1)
MINUS_ONE = Constant(-1)
E = Constant(np.e)
PI = Constant(np.pi)


def const(x: Any) -> Constant:
    return Constant(x)
