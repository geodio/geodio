import random
from abc import ABC
from typing import Dict

from src.cell.derivable import Derivable
from src.cell.operands.function import Function


class Functor(Function):
    def __init__(self, func_id, func, arity):
        super().__init__(arity, func)
        self.func_id = func_id
        self.func = func


class Functors(ABC):
    def __init__(self):
        self._functors: Dict[str, Functor] = {}

    def add_functor(self, functor: Functor):
        self._functors[functor.func_id] = functor

    def get_functor(self, func_id: str) -> Functor:
        return self._functors.get(func_id)

    def __getitem__(self, func_id: str) -> Functor:
        return self.get_functor(func_id)

    def __setitem__(self, func_id: str, functor: Functor):
        self.add_functor(functor)

    def __contains__(self, func_id: str) -> bool:
        return func_id in self._functors

    def __iter__(self):
        return iter(self._functors.values())

    def __len__(self):
        return len(self._functors)

    def __repr__(self):
        return f"Functors({list(self._functors.keys())})"

    def get_random_clone(self) -> Functor:
        if not self._functors:
            raise ValueError("No functors available to clone.")
        random_functor = random.choice(list(self._functors.values()))
        return random_functor.clone()


class CollectionBasedFunctors(Functors):
    def __init__(self, initial_functors=None):
        super().__init__()
        if initial_functors:
            for functor in initial_functors:
                self.add_functor(functor)
