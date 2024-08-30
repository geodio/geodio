"""
This module defines the `Operand` class, which serves as the foundational
class for all operations within the computational framework. It provides
a wide range of functionalities including arithmetic operations,
logical operations, derivative computation, and serialization.

The `Operand` class is abstract and is intended to be inherited by
specific operation classes such as addition, multiplication, and
conditional operations.

All operations derived from `Operand` can be composed, manipulated,
and evaluated within a computational graph, making this class a
core component of the system.

Each operand can be derived, cloned, serialized, and converted to a
Python-compatible string representation.
The class also provides an interface for creating mathematical and logical
operations dynamically through operator overloading.

Classes:
    Operand: Abstract base class for all operand types, providing essential
    methods for computation and manipulation.

Globals:
    GLOBAL_BUILTINS: A dictionary of functions that are dynamically populated
    to support various operations (e.g., addition, multiplication).
"""
import pickle
import sys
from abc import abstractmethod, ABCMeta
from typing import Optional, Dict, Any, Callable, Union

import core
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
    """
    The `Operand` class is an abstract base class for all operands in the computational graph.
    It provides a comprehensive interface for derivative computation, operand manipulation,
    and interaction with other operands via operator overloading.

    Inherits from:
        - Derivable: Provides methods for calculating derivatives.
        - WeightDerivable: Handles derivatives with respect to weights.
        - Hashable: Supports hash-based operations for caching and comparisons.

    Attributes:
        arity (int): The number of children (arguments) the operand expects.
        children (list): A list of child operands.
        error (int): An error value, initialized to the system's maximum integer value.
        id (int): A unique identifier for the operand instance.
    """

    def __init__(self, arity):
        """
       Initializes an Operand instance.

       Args:
           arity (int): The number of arguments (children) the operand will
           handle.

       """
        self.arity = arity
        self.children = []
        self.error = sys.maxsize
        self.id = 0

    @abstractmethod
    def __call__(self, args, meta_args: Optional[Dict[str, Any]] = None):
        """
        Evaluates the operand using the provided arguments.

        Args:
            args (Iterable): A list of arguments to evaluate the operand.
            meta_args (Optional[Dict[str, Any]]): Additional metadata arguments, if any.

        Returns:
            The result of evaluating the operand.

        Example:
            >>> op = SomeConcreteOperand(arity=2)
            >>> result = op([arg1, arg2])
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
        """
        Replaces an existing child operand with a new one.

        Args:
            old_child (Operand): The child operand to be replaced.
            new_child (Operand): The new operand to replace the old one.
        """
        i = self.children.index(old_child)
        self.children[i] = new_child

    @abstractmethod
    def clone(self) -> "Operand":
        """
        Creates a deep copy of the operand.

        Returns:
            Operand: A new instance of the operand with the same structure.

        Example:
            >>> op = SomeConcreteOperand(arity=2)
            >>> cloned_op = op.clone()
        """
        pass

    @abstractmethod
    def to_python(self) -> str:
        """
        Converts the operand to a Python-compatible string representation.

        Returns:
            str: A string that represents the operand in Python syntax.

        Example:
            >>> python_code = op.to_python()
        """
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
        """
        Computes the derivative of the operand.

        Args:
            index (int): The index with respect to which the derivative is computed.
            by_weights (bool): Flag indicating whether to derive by weights.

        Returns:
            Operand: The derived operand.
        """
        pass

    def d_w(self, dw):
        """
        Computes the derivative of the operand with respect to a weight.

        Args:
            dw (int): The index of the weight.

        Returns:
            Operand: The derived operand with respect to the weight.
        """
        return self.derive(dw, True)

    def to_bytes(self):
        """
        Serializes the operand to a byte stream.

        Returns:
            bytes: The serialized byte stream.

        Example:
            >>> from core.cell.operands import const
            >>> op = const(2)
            >>> byte_data = op.to_bytes()
        """
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data):
        """
        Deserializes a byte stream to an operand instance.

        Args:
            data (bytes): The byte stream to deserialize.

        Returns:
            Operand: The deserialized operand instance.

        Example:
            >>> from core.cell.operands import const
            >>> op = const(2)
            >>> byte_data = op.to_bytes()
            >>> des_op = Operand.from_bytes(byte_data)
        """
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

    def __and__(self, other):
        return GLOBAL_BUILTINS["and"](self, other)

    def __neg__(self):
        return GLOBAL_BUILTINS["not"](self)

    def __or__(self, other):
        return GLOBAL_BUILTINS["or"](self, other)

    def __gt__(self, other):
        return GLOBAL_BUILTINS["greater_than"](self, other)

    def __lt__(self, other):
        return GLOBAL_BUILTINS["less_than"](self, other)

    def __ge__(self, other):
        # TODO ADD DOC
        return GLOBAL_BUILTINS["greater_than_or_equal"](self, other)

    def __le__(self, other):
        return GLOBAL_BUILTINS["less_than_or_equal"](self, other)

    def equals(self, other):
        return GLOBAL_BUILTINS["equals"](self, other)

    def link(self, other: "Operand") -> "core.cell.Linker":
        """
        Links the current operand with another operand to form a composite
        function.

        Example:
            >>> from core.cell.operands import const, var
            >>> a = const(4)
            >>> b = (var(0) ** const(3)) + const(5)
            >>> x = a.link(b)
            >>> print(x([]))  # This will print 69 since 4 ** 3 + 5 is 69
            69

        Args:
            other (Operand): The operand to link with.

        Returns:
            Linker: The linked operand representing f(g(x)) where f is the
            current operand, and g is the other operand.
        """
        return GLOBAL_BUILTINS["link"](self, other)

    def __pow__(self, power: "Operand"):
        """
        Raises the operand to the power of another operand.

        Args:
            power (Operand): The exponent operand.

        Returns:
            Operand: The result of the power operation.

        Example:
            >>> from core.cell.operands import const
            >>> op1 = const(4.0)
            >>> op2 = const(2)
            >>> result = op1 ** op2
            >>> print(result([]))
            16.0
        """
        return GLOBAL_BUILTINS["pow"](self, power)

    T = property(lambda self: GLOBAL_BUILTINS["transpose"](self))
    """
    Property representing the transpose of the operand.

    Example:
        >>> from core.cell.operands import const
        >>> import numpy as np
        >>> op = const(np.arange(10).reshape(2, 5))
        >>> transposed_op = op.T
        >>> print(transposed_op([]))
        [[0 5]
         [1 6]
         [2 7]
         [3 8]
         [4 9]]
    """

    def hash_str(self):
        """
        Returns a string representation for hashing purposes.

        Returns:
            str: The hash string of the operand.

        Example:
            >>> from core.cell.operands import const
            >>> op = const(4) * const(5)
            >>> hash_str = op.hash_str()
            >>> print(hash_str)
            prod

        """
        return self.__name__

    def hash_tree(self) -> HashTree:
        """
        Returns a hash tree representation of the operand and its children.

        Returns:
            HashTree: The hash tree of the operand.
        """
        return HashNode(
            self.hash_str(),
            [kid.hash_tree() for kid in self.get_children()]
        )
