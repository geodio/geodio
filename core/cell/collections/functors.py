import random
from abc import ABC
from typing import Dict

from core.cell.collections.bank import Bank
from core.cell.operands.function import Function


class Functor(Function):
    def __init__(self, func_id, func, arity):
        super().__init__(arity, func)
        self.func_id = func_id
        self.func = func


class Functors(ABC, Bank[Functor]):
    def __init__(self):
        super().__init__()

    def add_functor(self, functor: Functor):
        self[functor.func_id] = functor


class CollectionBasedFunctors(Functors):
    def __init__(self, initial_functors=None):
        super().__init__()
        if initial_functors:
            for functor in initial_functors:
                self.add_functor(functor)
