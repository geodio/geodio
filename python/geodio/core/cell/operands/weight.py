from abc import abstractmethod, ABCMeta
from typing import Union

import numpy as np
from typing_extensions import TypeVar

from geodio.core.cell.operands.constant import ONE, ZERO
from geodio.core.cell.operands.operand import Operand


def adapt_shape_and_apply(_w, *args):
    """
    Adapts the shape of _w to match the shape of the first argument in args if they differ.
    If the first argument is a scalar, it ensures _w is also a scalar.
    Then applies the function func with _w and the rest of the arguments.

    Parameters:
    - func: The function to apply.
    - _w: The variable whose shape may need to be adapted.
    - *args: Additional arguments for the function func.

    Returns:
    - adapted weight and boolean, which is true if the weight has changed
    """
    if len(args) == 0:
        return _w, False
    # Get the first argument
    first_arg = args[0]
    # Check if the first argument is a scalar (int or float)
    if np.isscalar(first_arg):
        if np.isscalar(_w):
            # Both are scalars, no need to reshape
            return _w, False
        else:
            # The first argument is a scalar but _w is not, reduce _w to
            # a scalar
            return np.mean(np.array(_w)), True
    else:
        # The first argument is not a scalar, get its shape
        first_arg_shape = np.shape(first_arg)
        # Check the shape of _w
        w_shape = np.shape(_w)
        # If shapes differ, adapt the shape of _w
        if first_arg_shape != w_shape:
            try:
                return np.reshape(_w, first_arg_shape), True
            except ValueError:
                return np.zeros_like(first_arg_shape), True
        else:
            return _w, False


class LockedException(Exception):
    def __init__(self, msg: str):
        self.msg = f"Attempted to alter value of locked weight: {msg}"
        super().__init__(self.msg)


class AbsWeight(Operand, metaclass=ABCMeta):

    def __init__(self, adaptive_shape=False):
        super().__init__(0)
        self.w_index = 0
        self.adaptive_shape = adaptive_shape
        self._locked = False
        self.adapted = False

    @abstractmethod
    def set(self, weight: Union[np.ndarray, float, 't_weight']) -> None:
        pass

    @abstractmethod
    def get(self, **kwargs) -> Union[np.ndarray, float]:
        pass

    def __call__(self, args, meta_args=None):
        _w = self.get()
        if self.adaptive_shape and not self.adapted:
            _w, b = adapt_shape_and_apply(_w, *args)
            if b:
                self.set(_w)
            self.adapted = True
        return _w

    def __invert__(self):
        return None

    def get_weights_local(self):
        return [self]

    def set_weights(self, new_weights):
        if self._locked:
            raise LockedException(self.to_python())
        new_weight = new_weights[0]
        self.set(new_weight)

    def set_to_zero(self):
        _w = self.get()
        if isinstance(_w, np.ndarray):
            self.set(np.zeros_like(_w))
        else:
            self.set(0)

    def lock(self):
        """
        Locks the cell
        :return: None
        """
        self._locked = True

    def unlock(self):
        """
        Unlocks the cell
        :return: None
        """
        self._locked = False

    is_locked = property(lambda self: self._locked)

    def __eq__(self, other):
        if isinstance(other, AbsWeight):
            return (self.w_index == other.w_index and self.get() ==
                    other.get() and self._locked == other._locked and
                    self.adapted == other.adapted)

    def subscribe_to_graph(self, graph_wrapper):
        super().subscribe_to_graph(graph_wrapper)
        graph_wrapper.add_weight(self.graph_id, Tensor(self.get()))
        return self.graph_id

    def get_operand_type(self):
        return geodio.geodio_bindings.OperandType.Weight


class Weight(AbsWeight):
    def __init__(self, weight: Union[np.ndarray, float] = 0.0,
                 adaptive_shape=False):
        super().__init__(adaptive_shape)
        self.__weight = weight

    def d(self, var_index):
        if isinstance(self.get(), np.ndarray):
            return Weight(np.zeros_like(self.get()))
        return ZERO

    def d_w(self, dw):
        if isinstance(self.get(), (np.ndarray, list)):
            return Weight(np.ones_like(self.get())) \
                if self.w_index == dw \
                else Weight(np.zeros_like(self.get()))
        return ONE if self.w_index == dw else ZERO

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def clone(self):
        weight_clone = self.__weight
        if isinstance(self.__weight, np.ndarray):
            weight_clone = np.array(self.__weight)
        w_clone = Weight(weight_clone, self.adaptive_shape)
        w_clone.w_index = self.w_index
        return w_clone

    def to_python(self) -> str:
        return str(self.__weight)

    def get(self, **kwargs):
        return self.__weight

    def set(self, weight: Union[np.ndarray, float, 't_weight']):
        if isinstance(weight, AbsWeight):
            self.__weight = weight.get()
        else:
            self.__weight = weight


ZERO_WEIGHT = Weight(0)
t_weight = TypeVar('t_weight', bound=AbsWeight)


class ShapedWeight(AbsWeight):
    def __init__(self, shape, weight: Union[np.ndarray, float] = None):
        super().__init__(adaptive_shape=False)
        self.shape = shape
        if weight is None:
            self.__weight = np.zeros(shape)
        else:
            self.set(weight)

    def set(self, weight: Union[np.ndarray, float, 't_weight']):
        if isinstance(weight, AbsWeight):
            weight = weight.get()
        if isinstance(weight, np.ndarray):
            if weight.shape != self.shape:
                raise ValueError(f"Weight shape {weight.shape} does not "
                                 f"match required shape {self.shape}.")
            self.__weight = weight
        else:
            self.__weight = np.full(self.shape, weight)

    def get(self, **kwargs) -> np.ndarray:
        return self.__weight

    def d(self, var_index):
        derivative = ShapedWeight(self.shape, np.zeros(self.shape))
        derivative.lock()
        return derivative

    def d_w(self, dw):
        if self.w_index == dw:
            derivative = ShapedWeight(self.shape, np.ones(self.shape))
        else:
            derivative = ShapedWeight(self.shape, np.zeros(self.shape))
        derivative.lock()
        return derivative

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def clone(self) -> 'ShapedWeight':
        sw = ShapedWeight(self.shape, np.copy(self.__weight))
        sw.w_index = self.w_index
        return sw

    def to_python(self) -> str:
        return str(self.__weight.shape)
