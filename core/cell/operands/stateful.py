from abc import ABC
from typing import Union, TypeVar

import numpy as np

from core.cell.operands.constant import ZERO, ONE
from core.cell.operands.weight import AbsWeight, Weight, t_weight


class State(AbsWeight):

    def __init__(self, cell: 't_stateful', adaptive_shape=False):
        super().__init__(adaptive_shape)
        self.cell = cell
        self.cell.state_weight = self

    def d(self, var_index):
        if isinstance(self.cell.state, np.ndarray):
            return Weight(np.zeros_like(self.cell.state))
        return ZERO

    def d_w(self, dw):
        state = self.cell.state
        if isinstance(state, (np.ndarray, list)):
            return Weight(np.ones_like(state)) \
                if self.w_index == dw \
                else Weight(np.zeros_like(state))
        return ONE if self.w_index == dw else ZERO

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def clone(self):
        cloned_cell = self.cell.clone()
        w_clone = State(cloned_cell, self.adaptive_shape)
        w_clone.w_index = self.w_index
        return w_clone

    def to_python(self) -> str:
        return f"_state({str(self.cell.state)} ~ {str(self.cell.id)})"

    def set(self, weight: Union[np.ndarray, float, t_weight]) -> None:
        if isinstance(weight, AbsWeight):
            self.cell.state = weight.get()
        else:
            self.cell.state = weight

    def get(self) -> Union[np.ndarray, float]:
        return self.cell.state


class Stateful(ABC):
    def __init__(self):
        self.__current_state = self.state = 0.0
        self.state_weight = None
        self._previous_state = None
        self.__using_checkpoint = False

    def get_state_weight(self):
        if self.state_weight is None:
            self.state_weight = State(self)
        return self.state_weight

    def mark_checkpoint(self):
        self._previous_state = self.state

    checkpoint = property(lambda self: self._previous_state)

    def use_checkpoint(self):
        if self.__using_checkpoint:
            return
        self.__using_checkpoint = True
        self.__current_state = self.state
        self.state = self._previous_state

    def use_current(self):
        if not self.__using_checkpoint:
            return
        self.state = self.__current_state

    def update(self, new_state):
        self.__current_state = self.state = new_state

    def revert(self):
        """
        Revert the current state to the stored checkpoint.
        :return:
        """
        self.__using_checkpoint = False
        self.state = self._previous_state

    has_checkpoint = property(lambda self: self._previous_state is not None)


t_stateful = TypeVar('t_stateful', bound=Stateful)
