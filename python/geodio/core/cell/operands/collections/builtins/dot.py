import numpy as np

from geodio.core.cell.operands.collections.builtins.add import Add
from geodio.core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction, clean_number


class Dot(BuiltinBaseFunction):
    def __init__(self, children):
        super().__init__(children, "dot_prod", 2)

    def __call__(self, args, meta_args=None):
        a = self.children[0](args, meta_args)
        b = self.children[1](args, meta_args)

        r = clean_number(np.dot(a, b))
        return r

    def derive(self, index, by_weights=True):
        return Add(
            [
                Dot([self.children[0].derive(index, by_weights),
                     self.children[1]]),
                Dot([self.children[0], self.children[1].derive(index,
                                                               by_weights)])
            ],
            2
        )

    def clone(self) -> "Dot":
        return Dot([child.clone() for child in self.children])