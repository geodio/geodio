"""
Module containing built-in addition operation.
"""
from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction, clean_number


class Add(BuiltinBaseFunction):
    """
    Built-in addition operation.
    """
    def __init__(self, children, arity):
        super().__init__(children, f"add_{arity}", arity)

    def __call__(self, args, meta_args=None):
        return clean_number(
            sum([child(args, meta_args) for child in self.children]))

    def derive(self, index, by_weights=True):
        return Add(
            [child.derive(index, by_weights) for child in self.children],
            arity=self.arity)

    def clone(self) -> "Add":
        return Add([child.clone() for child in self.children], self.arity)

    def to_python(self) -> str:
        return " + ".join(child.__repr__() for child in self.children)
