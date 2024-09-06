from abc import ABC, abstractmethod

import numpy as np

from core.cell import Operand, matmul_of, transpose_of
from core.cell.operands import Function
from core.cell.train import BOO


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


class SigmoidActivation(ActivationFunction):
    def __init__(self, children, optimizer=None):
        super().__init__(children, optimizer)

        def d_sigmoid(z):
            x = 1 / (1 + np.exp(-z))
            derivative = x * (1 - x)
            return derivative

        self._derivative = d_sigmoid

    def actual_forward(self, x, meta_args=None):
        return 1 / (1 + np.exp(-x))

    def clone(self) -> "SigmoidActivation":
        return SigmoidActivation([child.clone for child in self.children],
                                 optimizer=self.optimizer.clone())

    def to_python(self) -> str:
        return "sigmoid"


class LinearActivation(ActivationFunction):
    def __init__(self, children, optimizer=None):
        super().__init__(children, optimizer)

        def d_linear(z):
            return np.ones_like(z)

        self._derivative = d_linear

    def actual_forward(self, x, meta_args=None):
        return x[0]

    def clone(self) -> "LinearActivation":
        return LinearActivation([child.clone for child in self.children],
                                optimizer=self.optimizer.clone())

    def to_python(self) -> str:
        return "linear"


class ReLUActivation(ActivationFunction):
    def __init__(self, children, optimizer=None):
        super().__init__(children, optimizer)

        def d_relu(z):
            return np.where(z > 0, 1, 0)

        self._derivative = d_relu

    def actual_forward(self, x, meta_args=None):
        return np.maximum(0, x)  # ReLU activation

    def clone(self):
        return ReLUActivation([child.clone for child in self.children],
                              optimizer=self.optimizer.clone())

    def to_python(self) -> str:
        return "relu"


def softmax(x):
    """
    Parameters

    x: input matrix of shape (m, d)
    where 'm' is the number of samples (in case of batch gradient descent of size m)
    and 'd' is the number of features
    """
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax_r = numerator / denominator
    return softmax_r


def d_softmax(x):
    """
    Parameters

    x: input matrix of shape (m, d)
    where 'm' is the number of samples (in case of batch gradient descent of size m)
    and 'd' is the number of features
    """
    if len(x.shape) == 1:
        x = np.array(x).reshape(1, -1)
    else:
        x = np.array(x)
    m, d = x.shape
    a = softmax(x)
    tensor1 = np.einsum('ij,ik->ijk', a, a)
    tensor2 = np.einsum('ij,jk->ijk', a, np.eye(d, d))
    return tensor2 - tensor1


class SoftmaxActivation(ActivationFunction):
    def __init__(self, children, optimizer=None):
        super().__init__(children, optimizer)

        self._derivative = d_softmax

    def actual_forward(self, x, meta_args=None):
        return softmax(x)

    def clone(self) -> "SoftmaxActivation":
        return SoftmaxActivation([child.clone for child in self.children],
                                 optimizer=self.optimizer.clone())

    def to_python(self) -> str:
        return "softmax"
