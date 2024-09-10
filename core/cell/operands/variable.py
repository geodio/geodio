from abc import ABCMeta, abstractmethod
from typing import Optional, List, Dict, Any

import numpy as np

from core.cell.operands.constant import ONE, ZERO, Constant
from core.cell.operands.operand import Operand
from core.cell.math.hashy import HashLeaf
from core.cell.math.backpropagation import Backpropagatable
from core.cell.operands.weight import AbsWeight


class BaseVariable(Operand, metaclass=ABCMeta):
    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def __call__(self, args, meta_args=None):
        return args[self.value]

    def d(self, var_index) -> Optional[Operand]:
        return ONE if self.value == var_index else ZERO

    def d_w(self, var_index) -> Optional[Operand]:
        return ZERO

    def __invert__(self):
        return None  # Variables do not have an inverse

    @abstractmethod
    def clone(self) -> "BaseVariable":
        pass

    def to_python(self) -> str:
        return f"x[{self.value}]"

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def hash_tree(self):
        return HashLeaf(self.value)


class AdaptiveConstant(BaseVariable):
    def __init__(self, value, constant):
        super().__init__(value)
        self.__constant = constant

    def clone(self) -> "AdaptiveConstant":
        return AdaptiveConstant(self.value, self.__constant)

    def __call__(self, args, meta_args=None):
        arg = args[self.value]
        if isinstance(arg, np.ndarray):
            return np.ones_like(arg) * self.__constant
        return self.__constant

    def to_python(self) -> str:
        return f"ADACON_{self.value}[{self.__constant}]"


class Variable(BaseVariable):
    def __init__(self, value):
        super().__init__(value)
        self.one = AdaptiveConstant(value, 1.0)
        self.zero = AdaptiveConstant(value, 0.0)

    def clone(self) -> "Variable":
        return Variable(self.value)

    def d(self, var_index) -> Optional[Operand]:
        return self.one if self.value == var_index else self.zero

    def d_w(self, var_index) -> Optional[Operand]:
        return self.zero


class BackpropagatableVariable(Variable, Backpropagatable):

    def __init__(self):
        super().__init__(0)
        self.one = AdaptiveConstant(0, 1.0)
        self.zero = AdaptiveConstant(0, 0.0)

    def clone(self) -> "BackpropagatableVariable":
        return BackpropagatableVariable()

    def backpropagation(self, dx: np.ndarray, meta_args=None) -> np.ndarray:
        return dx

    def forward(self, x: np.ndarray, meta_args=None) -> np.ndarray:
        return x

    def get_local_gradients(self):
        return []


class MetaOperand(Operand, metaclass=ABCMeta):
    pass


class MetaVariable(MetaOperand):
    def __init__(self, meta_id):
        super().__init__(0)
        self.__meta_id = meta_id

    def clone(self) -> "MetaVariable":
        return MetaVariable(self.__meta_id)

    def __call__(self, args, meta_args=None):
        if meta_args is None or meta_args.get(self.__meta_id) is None:
            raise ValueError(f"{self.__meta_id} is undefined")
        result = meta_args[self.__meta_id]
        while isinstance(result, MetaVariable) \
                or isinstance(result, Constant)\
                or isinstance(result, AbsWeight):
            result = result(args, meta_args)
        return result

    def __invert__(self):
        pass

    def to_python(self) -> str:
        return f"M[{self.__meta_id}]"

    def derive(self, index, by_weights=True):
        # TODO
        return NotImplemented

    meta_id = property(lambda self: self.__meta_id)


class MetaArgumented(MetaOperand):
    def __init__(self, params: List[str], function: Operand):
        super().__init__(len(params))
        self.params = params
        self.function = function

    def clone(self) -> "MetaArgumented":
        return MetaArgumented(self.params.copy(), self.function.clone())

    def __call__(self, args, meta_args=None):
        if meta_args is None:
            meta_args = {}
        else:
            meta_args = meta_args.copy()
        meta_args.update(zip(self.params, args))
        return self.function(args, meta_args=meta_args)

    def __invert__(self):
        pass

    def to_python(self) -> str:
        return f"op {self.function} ({', '.join(self.params)})"

    def derive(self, index, by_weights=True):
        # TODO
        return NotImplemented


def var(value: int) -> Variable:
    return Variable(value)


def b_var() -> BackpropagatableVariable:
    return BackpropagatableVariable()


class MetaAssignment(MetaOperand):
    def __init__(self, key: str, value: Operand):
        super().__init__(1)
        self.key = key
        self.value = value

    def clone(self) -> "MetaAssignment":
        return MetaAssignment(self.key, self.value.clone())

    def __call__(self, args, meta_args=None):
        meta_args[self.key] = self.value
        return self.value

    def __invert__(self):
        return ~self.value

    def to_python(self) -> str:
        return f"{self.key} = {self.value}"

    def derive(self, index, by_weights=True):
        return self.value.derive(index, by_weights=by_weights)


class MetaCall(MetaOperand):
    def __init__(self, meta_function_name: str, args: List[Operand]):
        super().__init__(1)
        self.mfn = meta_function_name
        self.children = args

    def __call__(self, args, meta_args: Optional[Dict[str, Any]] = None):
        arguments = [
            children(args, meta_args=meta_args) for children in self.children
        ]
        function = meta_args[self.mfn]
        result = function(arguments, meta_args=meta_args)
        return result

    def __invert__(self):
        return NotImplemented

    def clone(self) -> "Operand":
        return NotImplemented

    def to_python(self) -> str:
        str_children = [child.to_python() for child in self.children]
        return f"{self.mfn}({', '.join(str_children)})"

    def derive(self, index, by_weights=True):
        return NotImplemented
