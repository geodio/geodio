from typing import List

import numpy as np

from core.cell import Linker, LinearTransformation, ShapedWeight
from core.cell import OptimizableOperand
from core.cell.math.backpropagation import Backpropagatable
from core.organism.activation_function import ActivationFunction


class Node(OptimizableOperand, Backpropagatable):
    def __init__(self, arity, dim_in, dim_out, activ_fun: ActivationFunction,
                 optimizer=None, scalar_output=False):
        super().__init__(arity, optimizer)
        self.db = None
        self.dW = None
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
        self.scalar_output = scalar_output

    def __call__(self, x, meta_args=None):
        try:
            try:
                ind = x[0]
            except IndexError:
                ind = x
            if np.isscalar(ind):
                ind = np.array([ind])
            self.input_data = ind
            self.forward(self.input_data)
        except ValueError:
            return self.activated_output

    def forward(self, x: np.ndarray, meta_args=None) -> np.ndarray:
        broadcast_bias = self.bias.get()
        if np.ndim(x) > 1:
            broadcast_bias = broadcast_bias[:, np.newaxis]
        self.z = self.weight.get() @ x + broadcast_bias
        self.activated_output = self.activ_fun([self.z])
        if self.scalar_output:
            self.activated_output = self.activated_output[0]
        return self.activated_output

    def backpropagation(self, dx: np.ndarray, meta_args=None) -> np.ndarray:
        dz = self.activ_fun.backpropagation(dx)
        dr = dz.copy()
        self.db = np.sum(dz, axis=1).reshape(-1, 1)
        self.dW = np.matmul(dr, self.input_data.T)
        dx = np.matmul(self.weight.get().T, dr)
        return dx

    def get_gradients(self) -> List[np.ndarray]:
        return [self.dW, self.db]

    def get_children(self):
        return [self.weight, self.bias]

    def to_python(self) -> str:
        return self.activ_fun.to_python() + "(\n" + (
                "\t" + self.weight.to_python() + " * x + " +
                self.bias.to_python()
        ) + "\n)"

    def derive_uncached(self, index, by_weights=True):
        z_function = LinearTransformation(self.dim_in, self.dim_out)
        z_function.weight = self.weight
        z_function.bias = self.bias
        flag_original = self.dim_out == 1
        link = Linker(self.activ_fun.clone(), z_function, (self.dim_in,),
                      flag_original=flag_original)
        unchained = link.derive(index, by_weights)
        return unchained

    def clone(self):
        cloned = Node(self.arity, self.dim_in, self.dim_out,
                      self.activ_fun.clone(),
                      self.optimizer.clone() if self.optimizer else None)
        cloned.weight = self.weight.clone()
        cloned.bias = self.bias.clone()
        return cloned
