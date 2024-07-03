from typing import Union

import numpy as np

from src.cell.operands.constant import ONE, ZERO
from src.cell.operands.operand import Operand


class Weight(Operand):
    def __init__(self, weight: Union[np.ndarray, float] = 0.0,
                 adaptive_shape=False):
        super().__init__(0)
        self.w_index = 0
        self.weight = weight
        self.adaptive_shape = adaptive_shape

    def __call__(self, args):
        if self.adaptive_shape:
            first_arg = args[0]
            np_out = isinstance(first_arg, np.ndarray)
            np_in = isinstance(self.weight, np.ndarray)
            if (
                    np_out and
                    (np_in and first_arg.shape() != self.weight.shape()) or
                    (not np_in)
            ):
                self.weight = np.zeros_like(first_arg)
            elif not np_out and np_in:
                self.weight = 0.0
        return self.weight

    def d(self, var_index):
        if isinstance(self.weight, np.ndarray):
            return Weight(np.zeros_like(self.weight))
        return ZERO

    def d_w(self, dw):
        if isinstance(self.weight, (np.ndarray, list)):
            return Weight(np.ones_like(self.weight)) \
                if self.w_index == dw \
                else Weight(np.zeros_like(self.weight))
        return ONE if self.w_index == dw else ZERO

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def __invert__(self):
        return None

    def get_weights(self):
        return [self]

    def set_weights(self, new_weights):
        new_weight = new_weights[0]
        if isinstance(new_weight, Weight):
            self.weight = new_weights[0].weight
        else:
            self.weight = new_weight

    def set_to_zero(self):
        if isinstance(self.weight, np.ndarray):
            self.weight = np.zeros_like(self.weight)
        else:
            self.weight = 0

    def clone(self):
        weight_clone = self.weight
        if isinstance(self.weight, np.ndarray):
            weight_clone = np.array(self.weight)
        w_clone = Weight(weight_clone, self.adaptive_shape)
        w_clone.w_index = self.w_index
        return w_clone

    def to_python(self) -> str:
        return str(self.weight)


ZERO_WEIGHT = Weight(0)
