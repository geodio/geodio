from abc import ABCMeta, abstractmethod
from typing import Optional, List, Dict

import numpy as np

from core.cell.collections.builtin_functors import Sub
from core.cell.operands.operand import Operand
from core.cell.operands.constant import ONE, ZERO


class BaseVariable(Operand, metaclass=ABCMeta):
    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def __call__(self, args, meta_args=None):
        return args[self.value]

    def d(self, var_index) -> Optional[Operand]:
        return ONE if self.value == var_index else ZERO

    def d_w(self, var_index) -> Optional[Operand]:
        return Sub([self, self])

    def __invert__(self):
        return None  # Variables do not have an inverse

    @abstractmethod
    def clone(self) -> "BaseVariable":
        pass

    def to_python(self) -> str:
        return f"x[{self.value}]"

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)


class AdaptiveConstant(BaseVariable):
    def __init__(self, value, constant):
        super().__init__(value)
        self.__constant = constant

    def clone(self) -> "AdaptiveConstant":
        return AdaptiveConstant(self.value, self.__constant)

    def __call__(self, args, meta_args=None):
        arg = args[self.value]
        if isinstance(arg, np.ndarray):
            return np.ones_like(arg) * self.__constant
        return self.__constant

    def to_python(self) -> str:
        return f"ADACON_{self.value}[{self.__constant}]"


class Variable(BaseVariable):
    def __init__(self, value):
        super().__init__(value)
        self.one = AdaptiveConstant(value, 1.0)
        self.zero = AdaptiveConstant(value, 0.0)

    def clone(self) -> "Variable":
        return Variable(self.value)

    def d(self, var_index) -> Optional[Operand]:
        return self.one if self.value == var_index else self.zero

    def d_w(self, var_index) -> Optional[Operand]:
        return self.zero


class MetaVariable(Operand):
    def __init__(self, meta_id):
        super().__init__(0)
        self.__meta_id = meta_id

    def clone(self) -> "MetaVariable":
        return MetaVariable(self.__meta_id)

    def __call__(self, args, meta_args=None):
        return meta_args[self.__meta_id]

    def __invert__(self):
        pass

    def to_python(self) -> str:
        return f"M[{self.__meta_id}]"

    def derive(self, index, by_weights=True):
        # TODO
        return

    meta_id = property(lambda self: self.__meta_id)