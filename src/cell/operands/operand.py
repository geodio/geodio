import pickle
import sys
from abc import ABC, abstractmethod
from typing import Optional

from src.math.derivable import Derivable, WeightDerivable


class Operand(ABC, Derivable, WeightDerivable):

    def __init__(self, arity):
        self.arity = arity
        self.children = []
        self.fitness = None

    @abstractmethod
    def __call__(self, args):
        """
        Calling the operand.

        :param args: of length self.arity
        :return: the result of the applied operand
        """
        pass

    def d(self, var_index) -> "Optional[Operand]":
        """
        Derivative of the operand.

        :param var_index: index of the variable in regard to which the
        derivative is applied
        :return: Operand representing the derivative of this operand, or None
        if the derivative could not be computed.
        """
        return self.derive(var_index, False)

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
        return []

    def set_weights(self, new_weights):
        pass

    @abstractmethod
    def derive(self, index, by_weights=True):
        pass

    def d_w(self, dw):
        return self.derive(dw, True)

    def to_bytes(self):
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data):
        return pickle.loads(data)

    def get_fit(self):
        return self.fitness if self.fitness is not None else sys.maxsize
