from abc import ABC, abstractmethod

import numpy as np

from core.cell.operands.function import Function, PassThrough
from core.cell.operands.operand import Operand
from core.cell.operands.variable import AdaptiveConstant


class ActivationFunction(Operand, ABC):

    def __init__(self):
        super().__init__(1)

    def __invert__(self):
        pass

    def derive(self, index, by_weights=True) -> Operand:
        if by_weights or index != 0:
            return AdaptiveConstant(0, 0)
        return self.get_derivative()

    @abstractmethod
    def clone(self) -> "ActivationFunction":
        pass

    @abstractmethod
    def get_derivative(self) -> Operand:
        pass


class SigmoidActivation(ActivationFunction):
    def __init__(self):
        super().__init__()

        def d_sigmoid(z):
            x = 1 / (1 + np.exp(-z[0]))
            deriv = x * (1 - x)
            # print(deriv, z)
            return deriv

        self._derivative = Function(1, d_sigmoid, [PassThrough(1)])

    def __call__(self, args):
        return 1 / (1 + np.exp(-args[0]))

    def clone(self) -> "SigmoidActivation":
        return SigmoidActivation()

    def to_python(self) -> str:
        return "sigmoid"

    def get_derivative(self) -> Operand:
        return self._derivative


class ReLUActivation(ActivationFunction):
    def __init__(self):
        super().__init__()

        def d_relu(z):
            return np.where(z[0] > 0, 1, 0)

        self._derivative = Function(1, d_relu, [PassThrough(1)])

    def __call__(self, x):
        return np.maximum(0, x)  # ReLU activation

    def clone(self):
        return ReLUActivation()

    def get_derivative(self):
        return self._derivative

    def to_python(self) -> str:
        return "relu"
