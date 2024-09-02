from abc import ABC, abstractmethod

import numpy as np

from core.cell import AdaptiveConstant, Operand, OptimizableOperand, \
    Backpropagatable
from core.cell.operands import Function, PassThrough


class ActivationFunction(OptimizableOperand, Backpropagatable, ABC):

    def __init__(self):
        super().__init__(1)
        self.X = None

    def __invert__(self):
        pass

    def __call__(self, x, meta_args=None):
        args = x[0]
        return self.forward(args, meta_args=meta_args)

    def derive_uncached(self, index, by_weights=True) -> Operand:
        if by_weights or index != 0:
            return AdaptiveConstant(0, 0)
        return self.get_derivative()

    @abstractmethod
    def clone(self) -> "ActivationFunction":
        pass

    @abstractmethod
    def get_derivative(self) -> Operand:
        pass

    def backpropagation(self, dz: np.ndarray, meta_args=None) -> np.ndarray:
        f_prime = self.d(0)([self.X])

        dx = dz * f_prime
        return dx

    def forward(self, x, meta_args=None):
        self.X = x
        return self.actual_forward(x, meta_args=meta_args)

    @abstractmethod
    def actual_forward(self, x, meta_args=None) -> np.ndarray:
        pass

    def get_gradients(self):
        return []


class SigmoidActivation(ActivationFunction):
    def __init__(self):
        super().__init__()

        def d_sigmoid(z):
            x = 1 / (1 + np.exp(-z[0]))
            deriv = x * (1 - x)
            return deriv

        self._derivative = Function(1, d_sigmoid, [PassThrough(1)])

    def actual_forward(self, x, meta_args=None):
        return 1 / (1 + np.exp(-x))

    def clone(self) -> "SigmoidActivation":
        return SigmoidActivation()

    def to_python(self) -> str:
        return "sigmoid"

    def get_derivative(self) -> Operand:
        return self._derivative


class LinearActivation(ActivationFunction):
    def __init__(self):
        super().__init__()

        def d_linear(z):

            return np.ones_like(z[0])

        self._derivative = Function(1, d_linear, [PassThrough(1)])

    def actual_forward(self, x, meta_args=None):
        return x[0]

    def clone(self) -> "LinearActivation":
        return LinearActivation()

    def to_python(self) -> str:
        return "linear"

    def get_derivative(self) -> Operand:
        return self._derivative


class ReLUActivation(ActivationFunction):
    def __init__(self):
        super().__init__()

        def d_relu(z):
            return np.where(z[0] > 0, 1, 0)

        self._derivative = Function(1, d_relu, [PassThrough(1)])

    def actual_forward(self, x, meta_args=None):
        return np.maximum(0, x)  # ReLU activation

    def clone(self):
        return ReLUActivation()

    def get_derivative(self):
        return self._derivative

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
    softmax = numerator / denominator
    return softmax


def d_softmax(x):
    """
    Parameters

    x: input matrix of shape (m, d)
    where 'm' is the number of samples (in case of batch gradient descent of size m)
    and 'd' is the number of features
    """
    x = x[0]
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
    def __init__(self):
        super().__init__()

        self._derivative = Function(1, d_softmax, [PassThrough(1)])

    def actual_forward(self, x, meta_args=None):
        return softmax(x)

    def clone(self) -> "SoftmaxActivation":
        return SoftmaxActivation()

    def to_python(self) -> str:
        return "softmax"

    def get_derivative(self) -> Operand:
        return self._derivative
