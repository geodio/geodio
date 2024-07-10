from abc import ABC, abstractmethod, ABCMeta
from typing import Union

import numpy as np

from core.cell.operands.constant import ONE, ZERO
from core.cell.operands.operand import Operand


class AbsWeight(Operand, metaclass=ABCMeta):

    def __init__(self, adaptive_shape=False):
        super().__init__(0)
        self.w_index = 0
        self.adaptive_shape = adaptive_shape
        self._locked = False

    @abstractmethod
    def set(self, weight: Union[np.ndarray, float]) -> None:
        pass

    @abstractmethod
    def get(self) -> Union[np.ndarray, float]:
        pass

    def __call__(self, args):
        _w = self.get()
        if self.adaptive_shape:
            self.__adapt_shape(_w, args)
            _w = self.get()
        return _w

    def __adapt_shape(self, _w, args):
        try:
            if len(args) > 0:
                first_arg = args[0]
                np_out = isinstance(first_arg, np.ndarray)
                np_in = isinstance(_w, np.ndarray)
                if np_out:
                    self.set(np.ones_like(first_arg))
                elif not np_out and np_in:
                    self.set(1.0)
            self.adaptive_shape = False
        except TypeError:
            pass

    def __invert__(self):
        return None

    def get_weights(self):
        return [self]

    def set_weights(self, new_weights):
        new_weight = new_weights[0]
        self.set(new_weight)

    def set_to_zero(self):
        _w = self.get()
        if isinstance(_w, np.ndarray):
            self.set(np.zeros_like(_w))
        else:
            self.set(0)

    def lock(self):
        """
        Locks the cell
        :return: None
        """
        self._locked = True

    def unlock(self):
        """
        Unlocks the cell
        :return: None
        """
        self._locked = False

    is_locked = property(lambda self: self._locked)


class Weight(AbsWeight):
    def __init__(self, weight: Union[np.ndarray, float] = 0.0,
                 adaptive_shape=False):
        super().__init__(adaptive_shape)
        self.__weight = weight

    def d(self, var_index):
        if isinstance(self.get(), np.ndarray):
            return Weight(np.zeros_like(self.get()))
        return ZERO

    def d_w(self, dw):
        if isinstance(self.get(), (np.ndarray, list)):
            return Weight(np.ones_like(self.get())) \
                if self.w_index == dw \
                else Weight(np.zeros_like(self.get()))
        return ONE if self.w_index == dw else ZERO

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def clone(self):
        weight_clone = self.__weight
        if isinstance(self.__weight, np.ndarray):
            weight_clone = np.array(self.__weight)
        w_clone = Weight(weight_clone, self.adaptive_shape)
        w_clone.w_index = self.w_index
        return w_clone

    def to_python(self) -> str:
        return str(self.__weight)

    def get(self):
        return self.__weight

    def set(self, weight: Union[np.ndarray, float]):
        if isinstance(weight, AbsWeight):
            self.__weight = weight.get()
        else:
            self.__weight = weight


ZERO_WEIGHT = Weight(0)
