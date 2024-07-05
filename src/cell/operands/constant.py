import numpy as np

from src.cell.operands.operand import Operand


class Constant(Operand):
    def __invert__(self):
        return self

    def clone(self) -> "Operand":
        return self

    def to_python(self) -> str:
        return f"{self.__value}"

    def __init__(self, weight):
        super().__init__(weight)
        self.__value = weight

    def __call__(self, x):
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


ZERO = Constant(0.0)
ONE = Constant(1)
MINUS_ONE = Constant(-1)
E = Constant(np.e)
PI = Constant(np.pi)
