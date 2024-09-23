import numpy as np

from core.cell.operands.collections.basefunctions import BaseFunction
from core.cell.operands.utility import verify_equal_children


def clean_number(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        if x is None or str(x) == 'nan' or str(x) == 'inf' or np.isinf(x):
            return 0
    except Exception:
        return 0
    return x


class BuiltinBaseFunction(BaseFunction):
    def __init__(self, children, func_id, arity):
        self.__name__ = func_id
        super().__init__(func_id, None, arity)
        self.children = children
        self.value = self

    def __eq__(self, other):
        if isinstance(other, BuiltinBaseFunction):
            return self.func_id == other.func_id and verify_equal_children(
                self, other)
        return False
