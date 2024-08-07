from typing import Iterable

import numpy as np

from core.cell.operands.collections.builtins.matmul import matmul_of, \
    matmul_any
from core.cell.operands.collections.builtins.transpose import transpose_of
from core.cell.optim.optimizable import Operand, \
    BackpropagatableOperand


def get_w_ids(operand: Operand) -> Iterable[int]:
    weights = operand.get_weights_local()
    w_ids = [w.w_index for w in weights]
    return w_ids


class Linker(BackpropagatableOperand):

    def __init__(self, f: Operand, g: Operand, input_shape=0,
                 mark=False):
        super().__init__(g.arity)
        self.f = f
        self.g = g
        self.g_input = None
        self.f_input = None
        self.input_shape = input_shape
        self.mark = mark

    def __call__(self, args, meta_args=None):
        self.g_input = args
        self.f_input = [self.g(args, meta_args)]
        return self.f(self.f_input, meta_args)

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
            derivative = self.__derive_unchained_g(by_weight, index)
        return derivative

    def __derive_chained_f(self, index):
        self_double = Linker(self.f.derive(index, True), self.g)
        return self_double

    def __derive_unchained_g(self, by_weight, index):
        chain = Linker(self.f.derive(0, False), self.g)
        chained = self.g.derive(index, by_weight)
        derivative = matmul_of(transpose_of(chain), chained)
        return derivative

    def clone(self):
        return Linker(self.f.clone(), self.g.clone())

    def to_python(self) -> str:
        return "[λX.[" + self.f.to_python() + "]" + self.g.to_python() + "]"

    def get_children(self):
        return [self.g, self.f]

    def backpropagation(self, dx: np.ndarray, meta_args=None) -> np.ndarray:
        # backpropagation through g
        g_w_ids = get_w_ids(self.g)
        dx_t = dx.T
        g_gradients = [
            matmul_any(dx_t, self.g.derive(w_id, True)(self.g_input))
            for w_id in g_w_ids
        ]
        dz = matmul_any(self.g.derive(0, False)(self.g_input), dx)
        dz_t = dz.T
        f_gradients = [
            matmul_any(dz_t, self.f.derive(w_id, True)(self.f_input))
            for w_id in g_w_ids
        ]
        dz = matmul_any(self.f.derive(0, False)(self.f_input), dz)
        return dz

    def get_gradients(self):
        return get_gradients(self)


def get_backpropagatable_gradients(f: BackpropagatableOperand,
                                   g: BackpropagatableOperand):
    gradients = g.get_gradients()
    gradients.extend(f.get_gradients())
    return gradients


def compute_gradients(linker: Linker):
    weights = linker.get_weights_local()
    w_ids = list(map(lambda x: x.w_index, weights))
    gradients = [linker.d_w(w_id) for w_id in w_ids]
    return gradients


def get_gradients(linker: Linker):
    f, g = linker.f, linker.g

    if isinstance(
            g, BackpropagatableOperand
    ) and isinstance(
        f, BackpropagatableOperand
    ):
        return get_backpropagatable_gradients(f, g)
    else:
        return compute_gradients(linker)
