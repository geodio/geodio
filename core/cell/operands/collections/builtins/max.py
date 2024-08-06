from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction


class Max(BuiltinBaseFunction):
    def __init__(self, children, arity):
        super().__init__(children, f"max_{arity}", arity)

    def __call__(self, args, meta_args=None):
        return max([child(args, meta_args) for child in self.children])

    def d(self, dx):
        # TODO the derivative is not correctly computed
        return Max([child.d(dx) for child in self.children], arity=self.arity)

    def clone(self) -> "Max":
        return Max([child.clone() for child in self.children], self.arity)