from abc import ABC

from core.cell.collections.bank import Bank
from core.cell.operands.function import Function


class BaseFunction(Function):
    def __init__(self, func_id, func, arity):
        super().__init__(arity, func)
        self.func_id = func_id
        self.func = func


class BaseFunctions(ABC, Bank[BaseFunction]):
    def __init__(self):
        super().__init__()

    def add_functor(self, functor: BaseFunction):
        self[functor.func_id] = functor


class CollectionBasedBaseFunctions(BaseFunctions):
    def __init__(self, initial_functors=None):
        super().__init__()
        if initial_functors:
            for functor in initial_functors:
                self.add_functor(functor)
