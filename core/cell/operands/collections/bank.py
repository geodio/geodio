import random
from typing import TypeVar, Generic, Dict

from core.cell.cell import t_cell
from core.freezing.freezer import freeze

T = TypeVar('T')


class Bank(Generic[T]):
    def __init__(self):
        self._ts: Dict[str, T] = {}

    def get_functor(self, t_id: str) -> T:
        return self._ts.get(t_id)

    def __getitem__(self, t_id: str) -> T:
        return self.get_functor(t_id)

    def __setitem__(self, t_id: str, t: T):
        self._ts[t_id] = t

    def __contains__(self, t_id: str) -> bool:
        return t_id in self._ts

    def __iter__(self):
        return iter(self._ts.values())

    def __len__(self):
        return len(self._ts)

    def __repr__(self):
        return f"{T}-Bank({list(self._ts.keys())})"

    def get_random_clone(self) -> T:
        if not self._ts:
            raise ValueError(f"No {T} available to clone.")
        random_t = random.choice(list(self._ts.values()))
        return random_t.clone()


class CellBank(Bank[t_cell]):

    def add_cell(self, cell: t_cell):
        self[freeze(cell)] = cell

