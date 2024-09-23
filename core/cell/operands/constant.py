from typing import Any

import numba
import numpy as np

from core.cell.operands.operand import Operand


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
        super().__init__(weight)
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


ZERO = Constant(0.0)
ONE = Constant(1)
MINUS_ONE = Constant(-1)
E = Constant(np.e)
PI = Constant(np.pi)


def const(x: Any) -> Constant:
    return Constant(x)