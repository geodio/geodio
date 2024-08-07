from typing import Iterable, List

import numpy as np

from core.cell.operands.collections.builtins.matmul import matmul_of, \
    matmul_any, Matmul
from core.cell.operands.collections.builtins.transpose import transpose_of
from core.cell.optim.optimizable import Operand, \
    BackpropagatableOperand


def get_w_ids(operand: Operand) -> Iterable[int]:
    weights = operand.get_weights_local()
    w_ids = [w.w_index for w in weights]
    return w_ids


def manually_backprop(f: Operand, dx: np.ndarray, f_input, meta_args=None) -> [
    List[np.ndarray], np.ndarray
]:
    f_w_ids = get_w_ids(f)
    dx_t = dx.T
    print("DX", dx.shape)
    print("F", f_input[0].shape)
    # TODO
    f_derivatives = [f.derive(w_id, True)(f_input, meta_args)
        for w_id in f_w_ids]
    for der in f_derivatives:
        print(der.shape)
    f_gradients = [
        dx_t @ der for der in f_derivatives
    ]
    dz = f.derive(0, False)(f_input, meta_args).T @ dx
    return f_gradients, dz


def backprop_operand(f: Operand, dx: np.ndarray, f_input, meta_args=None) -> [
    List[np.ndarray], np.ndarray
]:
    if isinstance(f, BackpropagatableOperand):
        dx = f.backpropagation(dx, meta_args=meta_args)
        f_gradients = f.get_gradients()
    else:
        f_gradients, dx = manually_backprop(f, dx, f_input,
                                            meta_args=meta_args)
    return f_gradients, dx


def spread_original_operand(f):
    if isinstance(f, Matmul):
        f.flag_original = True
    elif isinstance(f, Linker):
        f.flag_original = True
        f.spread_original()


class Linker(BackpropagatableOperand):

    def __init__(self, f: Operand, g: Operand, input_shape=0,
                 mark=False, flag_original=False):
        super().__init__(g.arity)
        self.gradients = []
        self.f_gradients = None
        self.g_gradients = None
        self.f = f
        self.g = g
        self.g_input = None
        self.f_input = None
        self.input_shape = input_shape
        self.mark = mark
        self.flag_original = flag_original
        if self.flag_original:
            self.spread_original()

    def spread_original(self):
        spread_original_operand(self.f)
        spread_original_operand(self.g)

    def __call__(self, args, meta_args=None):
        self.g_input = args
        self.f_input = [self.g(args, meta_args)]
        return self.f(self.f_input, meta_args)

    def forward(self, x, meta_args=None):
        pass

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
        derivative = matmul_of(transpose_of(chain), chained,
                               self.flag_original)
        return derivative

    def clone(self):
        return Linker(self.f.clone(), self.g.clone())

    def to_python(self) -> str:
        return "[λX.[" + self.f.to_python() + "]" + self.g.to_python() + "]"

    def get_children(self):
        return [self.g, self.f]

    def backpropagation(self, dx: np.ndarray, meta_args=None) -> np.ndarray:
        # backpropagation through f
        self.f_gradients, dx = backprop_operand(self.f, dx, self.f_input,
                                                meta_args)
        # backpropagation through g
        self.g_gradients, dz = backprop_operand(self.g, dx, self.g_input,
                                                meta_args)
        # gradients
        self.gradients = self.g_gradients + self.f_gradients
        return dz

    def get_gradients(self):
        return self.gradients
