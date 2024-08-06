import numpy as np

from core.cell.operands.collections.builtins.add import Add
from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction, clean_number


class Prod(BuiltinBaseFunction):
    def __init__(self, children):
        super().__init__(children, "prod", 2)

    def __call__(self, args, meta_args=None):
        a = self.children[0](args, meta_args)
        b = self.children[1](args, meta_args)
        try:
            return clean_number(a * b)
        except:
            a = a[:, np.newaxis]
            return clean_number(a * b)

    def derive(self, index, by_weights=True):
        return Add(
            [
                Prod([self.children[0].derive(index, by_weights),
                      self.children[1]]),
                Prod([self.children[0],
                      self.children[1].derive(index, by_weights)])
            ],
            2
        )

    def clone(self) -> "Prod":
        return Prod([child.clone() for child in self.children])

    def to_python(self) -> str:
        return str(self.children[0]) + " * " + str(self.children[1])
