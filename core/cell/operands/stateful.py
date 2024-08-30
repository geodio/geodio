from abc import ABC
from collections import deque
from typing import Union, TypeVar, Deque

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

    def get_children(self):
        return []

    def __eq__(self, other):
        if isinstance(other, State):
            return self.cell == other.cell
        return False


class Stateful(ABC):
    def __init__(self, max_checkpoints: int = 100):
        self.__current_state = self.state = 0.0
        self.state_weight = None
        # Deque for storing multiple states
        self._checkpoints: Deque = deque(maxlen=max_checkpoints)
        self.__using_checkpoint = False

    def get_state_weight(self) -> "State":
        if self.state_weight is None:
            self.state_weight = State(self)
        return self.state_weight

    def mark_checkpoint(self):
        # Append the current state to the checkpoints
        self._checkpoints.append(self.state)

    def get_checkpoint(self, index: int = -1):
        """
        Retrieve a checkpoint by index, defaulting to the last one (-1).
        """
        try:
            return self._checkpoints[index]
        except IndexError:
            raise ValueError(f"No checkpoint at index {index}")

    def use_checkpoint(self, index: int = -1):
        """
        Use a specific checkpoint by index, defaulting to the last one.
        """
        self.__using_checkpoint = True
        self.__current_state = self.state
        self.state = self.get_checkpoint(index)

    def use_current(self):
        if not self.__using_checkpoint:
            return
        self.state = self.__current_state
        self.__using_checkpoint = False

    def update(self, new_state):
        self.__current_state = self.state = new_state

    def revert(self, index: int = -1):
        """
        Revert the current state to a specific checkpoint.
        :param index: The index of the checkpoint to revert to, defaults to the
        last one.
        """
        self.__using_checkpoint = False
        self.state = self.get_checkpoint(index)

    def clear_checkpoints(self, keep_last: int = 1):
        """
        Clear all checkpoints except for the last 'keep_last' ones.
        :param keep_last: Number of checkpoints to keep from the end.
        """
        if keep_last >= len(self._checkpoints):
            return
        for _ in range(len(self._checkpoints) - keep_last):
            self._checkpoints.popleft()

    has_checkpoint = property(lambda self: len(self._checkpoints) > 0)


t_stateful = TypeVar('t_stateful', bound=Stateful)
