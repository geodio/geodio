from typing import Optional

from src.cell.operands.operand import Operand


class Function(Operand):
    def __init__(self, arity, value, children=None):
        super().__init__(arity)
        self.value = value
        self.children = children if children is not None else []

    def __call__(self, args):
        func_args = [child(args) for child in self.children]
        print(args)
        print(func_args)
        return self.value(*func_args)

    def d(self, var_index) -> Optional[Operand]:
        return None

    def __invert__(self):
        return None  # Replace with actual inversion logic

    def clone(self) -> "Function":
        return Function(self.arity, self.value, [child.clone() for child in
                                                 self.children])

    def to_python(self) -> str:
        args = [child.to_python() for child in self.children]
        return f"{self.value.__name__}({', '.join(args)})"
