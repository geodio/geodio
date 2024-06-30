from abc import ABC, abstractmethod
from typing import Optional

import numpy as np

from src.cell.derivable import Derivable


class Operand(ABC, Derivable):

    def __init__(self, arity):
        self.arity = arity
        self.children = []

    @abstractmethod
    def __call__(self, args):
        """
        Calling the operand.

        :param args: of length self.arity
        :return: the result of the applied operand
        """
        pass

    @abstractmethod
    def d(self, var_index) -> "Optional[Operand]":
        """
        Derivative of the operand.

        :param var_index: index of the variable in regard to which the
        derivative is applied
        :return: Operand representing the derivative of this operand, or None
        if the derivative could not be computed.
        """
        pass

    @abstractmethod
    def __invert__(self):
        """
        Inverse of the operand.

        :return: Operand representing the inverse of this operand, or None
        if the inverse could not be computed.
        """

    def add_child(self, child):
        self.children.append(child)

    def replace_child(self, old_child, new_child):
        i = self.children.index(old_child)
        self.children[i] = new_child

    @abstractmethod
    def clone(self) -> "Operand":
        pass

    @abstractmethod
    def to_python(self) -> str:
        pass

    def get_weights(self):
        return np.array([])

    def set_weights(self, new_weights):
        pass
