import numpy as np

from core.cell.operands.collections.activation_function. \
    base_activation_function import ActivationFunction


class ReLUActivation(ActivationFunction):
    def __init__(self, children, optimizer=None):
        super().__init__(children, optimizer)

        def d_relu(z):
            return np.where(z > 0, 1, 0)

        self._derivative = d_relu

    def actual_forward(self, x, meta_args=None):
        return np.maximum(0, x)  # ReLU activation

    def clone(self):
        return ReLUActivation([child.clone() for child in self.children],
                              optimizer=self.optimizer.clone())

    def to_python(self) -> str:
        return "relu"
