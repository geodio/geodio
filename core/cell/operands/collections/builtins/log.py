import numpy as np

from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction, clean_number
from core.cell.operands.collections.builtins.div import Div


class Log(BuiltinBaseFunction):
    def __init__(self, children):
        super().__init__(children, "log", 1)

    def __call__(self, args, meta_args=None):
        try:
            return clean_number(np.log(self.children[0](args, meta_args)))
        except:
            return 0

    def derive(self, index, by_weights=True):
        # d/dx (log(a)) = 1 / a * d/dx(a)
        a = self.children[0]
        return Div([
            a.derive(index, by_weights),
            a
        ])

    def clone(self) -> "Log":
        return Log([child.clone() for child in self.children])