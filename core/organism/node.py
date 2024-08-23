from typing import List

import numpy as np

from core.cell import Linker, LinearTransformation, ShapedWeight
from core.cell import OptimizableOperand
from core.organism.activation_function import ActivationFunction


class Node(OptimizableOperand):
    def __init__(self, arity, dim_in, dim_out, activ_fun: ActivationFunction,
                 optimizer=None):
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

    def __call__(self, x, meta_args=None):
        try:
            ind = x[0]
            biaas = self.bias.get()
            if np.ndim(ind) > 1:
                biaas = biaas[:, np.newaxis]
            self.z = self.weight.get() @ ind + biaas
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
