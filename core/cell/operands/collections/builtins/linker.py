from itertools import chain

import numpy as np

from core.cell.operands.collections.builtins.matmul import matmul_of, \
    Matmul
from core.cell.operands.collections.builtins.transpose import transpose_of
from core.cell.optim.optimizable import Operand, OptimizableOperand


def spread_original_operand(f):
    if isinstance(f, Matmul):
        f.flag_original = True
    elif isinstance(f, Linker):
        f.flag_original = True
        f.spread_original()


class Linker(OptimizableOperand):

    def __init__(self, f: Operand, g: Operand, input_shape=0,
                 mark=False, flag_original=False):
        super().__init__(g.arity)
        self.gradients = []
        self.f = f
        self.g = g
        self.input_shape = input_shape
        self.mark = mark
        self.flag_original = flag_original
        if self.flag_original:
            self.spread_original()

    def spread_original(self):
        spread_original_operand(self.f)
        spread_original_operand(self.g)

    def __call__(self, args, meta_args=None):
        f_input = [self.g(args, meta_args)]
        return self.f(f_input, meta_args)

    def derive_uncached(self, index, by_weight=True):
        """
        (f(g(x)))' = f'(g(x)) * g'(x)
        :param index:
        :param by_weight:
        :return:
        """
        if by_weight and self.g.is_independent_of(index):
            derivative = self.__derive_chained_f(index)
        else:
            derivative = self.derive_unchained_g(by_weight, index)
        return derivative

    def __derive_chained_f(self, index):
        self_double = Linker(self.f.derive(index, True), self.g)
        return self_double

    def derive_unchained_g(self, by_weight, index):
        chain = Linker(self.f.derive(0, False), self.g)
        chained = self.g.derive(index, by_weight)
        derivative = matmul_of(transpose_of(chain), chained,
                               self.flag_original)
        return derivative

    def clone(self):
        return Linker(self.f.clone(), self.g.clone())

    def to_python(self) -> str:
        return "[λX.[" + self.f.to_python() + "]" + self.g.to_python() + "]"

    def get_children(self):
        return [self.g, self.f]
