from typing import Optional

from src.cell.operands.operand import Operand
from src.cell.operands.constant import ONE, ZERO


class Variable(Operand):
    def __init__(self, value):
        super().__init__(0)
        self.value = value

    def __call__(self, args):
        return args[self.value]

    def d(self, var_index) -> Optional[Operand]:
        return ONE if self.value == var_index else ZERO

    def __invert__(self):
        return None  # Variables do not have an inverse

    def clone(self) -> "Variable":
        return Variable(self.value, self.arity)

    def to_python(self) -> str:
        return f"x[{self.value}]"
