import numpy as np

from core.cell.operands.collections.activation_function. \
    base_activation_function import ActivationFunction


def d_linear(z):
    return np.ones_like(z)


class LinearActivation(ActivationFunction):
    def __init__(self, children, optimizer=None):
        super().__init__(children, optimizer)

        self._derivative = d_linear

    def actual_forward(self, x, meta_args=None):
        return x[0]

    def clone(self) -> "LinearActivation":
        return LinearActivation([child.clone() for child in self.children],
                                optimizer=self.optimizer.clone())

    def to_python(self) -> str:
        return "linear"
