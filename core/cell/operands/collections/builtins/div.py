import sys

from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction, clean_number
from core.cell.operands.collections.builtins.prod import Prod
from core.cell.operands.collections.builtins.sub import Sub


class Div(BuiltinBaseFunction):
    def __init__(self, children):
        super().__init__(children, "div", 2)

    def __call__(self, args, meta_args=None):
        try:
            up = self.children[0](args, meta_args)
            down = self.children[1](args, meta_args)
        except IndexError:
            return 0.0
        if down == 0:
            if up >= 0:
                return sys.maxsize
            return -sys.maxsize
        return clean_number(up / down)

    def derive(self, index, by_weights=True):
        # d/dx (a / b) = (b * d/dx(a) - a * d/dx(b)) / (b^2)
        a, b = self.children[0], self.children[1]
        return Div.actual_derivative(a, b, a.derive(index, by_weights),
                                     b.derive(index, by_weights))

    @staticmethod
    def actual_derivative(a, b, a_d, b_d):
        return Div([
            Sub([
                Prod([b, a_d]),
                Prod([a, b_d])
            ]),
            Prod([b,b])
        ])

    def clone(self) -> "Div":
        return Div([child.clone() for child in self.children])

    def to_python(self) -> str:
        return ("(" + str(self.children[0]) + ") / (" + str(self.children[1])
                + ")")