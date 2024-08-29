from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction


class Condition(BuiltinBaseFunction):
    def __init__(self, children):
        super().__init__(children, "condition", len(children))

    def __call__(self, args, meta_args=None):
        raise NotImplementedError("Boolean operations must implement __call__")

    def clone(self) -> "Condition":
        return self.__class__([child.clone() for child in self.children])

    def to_python(self) -> str:
        raise NotImplementedError(
            "Boolean operations must implement to_python")

    def derive(self, index, by_weights=True):
        raise NotImplementedError(
            "Boolean operations do not support differentiation")
