import numpy as np

from core.cell.operands.collections.activation_function. \
    base_activation_function import ActivationFunction


def d_sigmoid(z):
    x = 1 / (1 + np.exp(-z))
    derivative = x * (1 - x)
    return derivative


class SigmoidActivation(ActivationFunction):
    def __init__(self, children, optimizer=None):
        super().__init__(children, optimizer)
        self._derivative = d_sigmoid

    def actual_forward(self, x, meta_args=None):
        return 1 / (1 + np.exp(-x))

    def clone(self) -> "SigmoidActivation":
        return SigmoidActivation([child.clone() for child in self.children],
                                 optimizer=self.optimizer.clone())

    def to_python(self) -> str:
        return "sigmoid"
