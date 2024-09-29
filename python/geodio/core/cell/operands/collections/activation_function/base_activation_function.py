from abc import ABC, abstractmethod

import numpy as np

from geodio.core.cell.operands.operand import Operand
from geodio.core.cell.train.boo import BOO
from geodio.core.cell.operands.collections.builtins.matmul import matmul_of
from geodio.core.cell.operands.collections.builtins.transpose import transpose_of
from geodio.core.cell.operands.function import Function


class ActivationFunction(BOO, ABC):

    def __init__(self, children, optimizer=None):
        super().__init__(children, optimizer)
        self.X = None
        self._derivative = None

    def derive_uncached(self, index, by_weights=True) -> Operand:
        local_derivative = self.get_local_derivative()
        chained = self.children[0].derive(index, by_weights)
        derivative = matmul_of(transpose_of(local_derivative), chained)
        return derivative

    @abstractmethod
    def clone(self) -> "ActivationFunction":
        pass

    def get_local_derivative(self) -> Operand:
        return Function(1, self._derivative, self.children)

    def backpropagation(self, dz: np.ndarray, meta_args=None) -> np.ndarray:
        f_prime = self._derivative(self.X)
        dx = dz * f_prime
        dx = self.children[0].backpropagation(dx, meta_args)
        return dx

    def forward(self, x, meta_args=None):
        self.X = x
        return self.actual_forward(x, meta_args=meta_args)

    @abstractmethod
    def actual_forward(self, x, meta_args=None) -> np.ndarray:
        pass

    def get_local_gradients(self):
        return []
