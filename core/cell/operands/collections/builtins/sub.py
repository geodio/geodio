from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction, clean_number


class Sub(BuiltinBaseFunction):
    def __init__(self, children):
        super().__init__(children, "sub", 2)

    def __call__(self, args, meta_args=None):
        try:
            return clean_number(
                self.children[0](args, meta_args) - self.children[1](args,
                                                                     meta_args)
            )
        except IndexError:
            return 0

    def derive(self, index, by_weights=True):
        return Sub([self.children[0].derive(index, by_weights),
                    self.children[1].derive(index, by_weights)])

    def clone(self) -> "Sub":
        return Sub([child.clone() for child in self.children])

    def to_python(self) -> str:
        return str(self.children[0]) + " - " + str(self.children[1])