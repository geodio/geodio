from typing import Union

import numpy as np

from core.cell.operands.collections.builtins import Linker
from core.cell.operands.function import Function, PassThrough
from core.cell.operands.weight import AbsWeight, t_weight
from core.cell.optim.optimizable import OptimizableOperand
from core.organism.activation_function import ActivationFunction


class ShapedWeight(AbsWeight):
    def __init__(self, shape, weight: Union[np.ndarray, float] = None):
        super().__init__(adaptive_shape=False)
        self.shape = shape
        if weight is None:
            self.__weight = np.zeros(shape)
        else:
            self.set(weight)

    def set(self, weight: Union[np.ndarray, float, 't_weight']):
        if isinstance(weight, AbsWeight):
            weight = weight.get()
        if isinstance(weight, np.ndarray):
            if weight.shape != self.shape:
                raise ValueError(f"Weight shape {weight.shape} does not "
                                 f"match required shape {self.shape}.")
            self.__weight = weight
        else:
            self.__weight = np.full(self.shape, weight)

    def get(self) -> np.ndarray:
        return self.__weight

    def d(self, var_index):
        derivative = ShapedWeight(self.shape, np.zeros(self.shape))
        derivative.lock()
        return derivative

    def d_w(self, dw):
        if self.w_index == dw:
            derivative = ShapedWeight(self.shape, np.ones(self.shape))
        else:
            derivative = ShapedWeight(self.shape, np.zeros(self.shape))
        derivative.lock()
        return derivative

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def clone(self) -> 'ShapedWeight':
        sw = ShapedWeight(self.shape, np.copy(self.__weight))
        sw.w_index = self.w_index
        return sw

    def to_python(self) -> str:
        return str(self.__weight.shape)


class LinearTransformation(OptimizableOperand):
    def __init__(self, dim_in, dim_out, optimizer=None):
        super().__init__(arity=1, optimizer=optimizer)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = ShapedWeight((dim_out, dim_in),
                                   np.random.randn(dim_out, dim_in))
        self.bias = ShapedWeight((dim_out,), np.zeros(dim_out))

    def __call__(self, args, meta_args=None):
        X = np.array(args[0])
        try:
            z = np.dot(self.weight.get(), X)
        except ValueError:
            z = np.dot(self.weight.get(), X[0])
        try:
            r = z + self.bias.get()
        except ValueError:
            r = z + self.bias.get()[:, np.newaxis]
        return r

    def derive_unchained(self, index, by_weights=True):
        if by_weights:
            if index == self.weight.w_index:  # Derivative with respect to W
                return self._derive_w()
            elif index == self.bias.w_index:  # Derivative with respect to B
                return self._derive_b()
            else:
                sw = ShapedWeight(
                    (self.dim_out, self.dim_out),
                    np.zeros((self.dim_out, self.dim_out))
                )
                sw.lock()
                return sw
        else:  # Derivative with respect to X
            return self._derive_x()

    def _derive_w(self):
        # The derivative of W * X + B with respect to W is X.
        def dW(args):
            X = np.array(args[0])
            # Repeat X to match the shape of W
            return X  # np.tile(X, (self.dim_out, 1))

        return Function(1, dW, [PassThrough(1)])

    def _derive_x(self):
        # The derivative of W * X + B with respect to X is W.
        def dX(args):
            return self.weight.get()

        return Function(1, dX, [PassThrough(1)])

    def _derive_b(self):
        # The derivative of W * X + B with respect to B is 1.
        def dB(args):
            return np.ones(self.dim_out)

        return Function(1, dB, [PassThrough(1)])

    def clone(self):
        cloned = LinearTransformation(self.dim_in, self.dim_out,
                                      self.optimizer.clone() if self.optimizer else None)
        cloned.weight = self.weight.clone()
        cloned.bias = self.bias.clone()
        return cloned

    def to_python(self) -> str:
        return f"{self.weight.to_python()} * x + {self.bias.to_python()}"

    def get_children(self):
        return [self.weight, self.bias]


class Node(OptimizableOperand):
    def __init__(self, arity, dim_in, dim_out, activ_fun: ActivationFunction,
                 optimizer=None):
        super().__init__(arity, optimizer)
        self.arity = arity
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activ_fun: ActivationFunction = activ_fun

        self.weight = ShapedWeight(
            (dim_out, dim_in), np.random.randn(dim_out, dim_in)
        )
        self.bias = ShapedWeight(
            (dim_out,), np.zeros(dim_out)
        )

        self.input_data = None
        self.z = None
        self.activated_output = None
        self.output_dimensionality = dim_out

    def __call__(self, args, meta_args=None):
        try:
            ind = np.array(args[0])
            biaas = self.bias.get()
            if np.ndim(ind) > 1:
                biaas = biaas[:, np.newaxis]
            self.input_data = ind
            self.z = self.weight.get() @ self.input_data + biaas
            self.activated_output = self.activ_fun([self.z])
            return self.activated_output
        except ValueError:
            return self.activated_output

    def get_children(self):
        return [self.weight, self.bias]

    def to_python(self) -> str:
        return self.activ_fun.to_python() + "(\n" + (
                "\t" + self.weight.to_python() + " * x + " +
                self.bias.to_python()
        ) + "\n)"

    def derive_unchained(self, index, by_weights=True):
        z_function = LinearTransformation(self.dim_in, self.dim_out)
        z_function.weight = self.weight
        z_function.bias = self.bias
        link = Linker(self.activ_fun, z_function, (self.dim_in,))
        unchained = link.derive(index, by_weights)
        return unchained

    def clone(self):
        cloned = Node(self.arity, self.dim_in, self.dim_out,
                      self.activ_fun.clone(),
                      self.optimizer.clone() if self.optimizer else None)
        cloned.weight = self.weight.clone()
        cloned.bias = self.bias.clone()
        return cloned
