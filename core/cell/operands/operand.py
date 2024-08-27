import pickle
import sys
from abc import abstractmethod, ABCMeta
from typing import Optional, Dict, Any, Callable, Union

from core.cell.math.derivable import Derivable, WeightDerivable
from core.cell.math.hashy import Hashable, HashTree, HashNode

GLOBAL_BUILTINS: Dict[
    str,
    Union[
        Callable[["Operand"], "Operand"],
        Callable[["Operand", "Operand"], "Operand"],
    ]
] = {}


class Operand(Derivable, WeightDerivable, Hashable, metaclass=ABCMeta):

    def __init__(self, arity):
        self.arity = arity
        self.children = []
        self.error = sys.maxsize
        self.id = 0

    @abstractmethod
    def __call__(self, args, meta_args: Optional[Dict[str, Any]] = None):
        """
        Calling the operand.

        :param meta_args:
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

    def get_weights_local(self):
        return []

    def get_weights(self):
        weights = self.get_weights_local()
        self.update_weights_index(weights)
        return weights

    def set_weights(self, new_weights):
        pass

    def is_independent_of(self, index) -> bool:
        """
        Checks if the operand is independent of the
        weight corresponding to given index. The
        method assumes that ``get_weights`` has already
        been called.
        Args:
            index: given index of the weight

        Returns:
            true, if independent of the weight;
            false, otherwise

        """
        weights = self.get_weights_local()
        for weight in weights:
            if weight.w_index == index:
                return False
        return True

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

    def get_error(self):
        return self.error if self.error is not None else sys.maxsize

    def __str__(self):
        return self.to_python()

    def __repr__(self):
        return self.to_python()

    def update_weights_index(self, weights):
        for i, weight in enumerate(weights):
            weight.w_index = i

    def get_children(self):
        return self.children

    def __eq__(self, other: "Operand") -> bool:
        return self == other

    def __matmul__(self, other: "Operand"):
        return GLOBAL_BUILTINS["matmul"](self, other)

    def __mul__(self, other: "Operand"):
        """
        Operand multiplication.

        :param other: Operand to be multiplied.
        :return: Prod operand, that multiplies the self with other operand.
        """
        return GLOBAL_BUILTINS["prod"](self, other)

    def __truediv__(self, other: "Operand"):
        """
        Operand division.

        :param other: Operand to be divided.
        :return: Div operand, that divides the self with the other operand.
        """
        return GLOBAL_BUILTINS["div"](self, other)

    def __add__(self, other: "Operand"):
        """
        Operand addition.

        :param other: Operand to be added.
        :return: Add operand, that adds the other operand with the self.
        """
        return GLOBAL_BUILTINS["add"](self, other)

    def __radd__(self, other: "Operand"):
        return GLOBAL_BUILTINS["add"](self, other)

    def __sub__(self, other: "Operand"):
        return GLOBAL_BUILTINS["sub"](self, other)

    def link(self, other: "Operand"):
        return GLOBAL_BUILTINS["link"](self, other)

    def __pow__(self, power: "Operand"):
        return GLOBAL_BUILTINS["pow"](self, power)

    T = property(lambda self: GLOBAL_BUILTINS["transpose"](self))

    def hash_str(self):
        return self.__name__

    def hash_tree(self) -> HashTree:
        return HashNode(
            self.hash_str(),
            [kid.hash_tree() for kid in self.get_children()]
        )
