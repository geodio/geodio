import numpy as np

from core.cell.operands.constant import Constant
from core.cell.operands.collections.builtins.add import Add
from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction, clean_number
from core.cell.operands.collections.builtins.log import Log
from core.cell.operands.collections.builtins.prod import Prod


class Power(BuiltinBaseFunction):
    def __init__(self, children):
        super().__init__(children, "power", 2)

    def __call__(self, args, meta_args=None):
        base_func = self.children[0]
        exponent = self.children[1](args, meta_args)
        try:
            if exponent == 0:
                return 1.0
            return clean_number(
                np.power(0.0 + base_func(args, meta_args), exponent))
        except:
            return 0.0

    def derive(self, index, by_weights=True):
        base = self.children[0]
        exponent = self.children[1]
        base_dx = base.derive(index, by_weights)
        exponent_dx = exponent.derive(index, by_weights)

        # d/dx (a^b) = b * a^(b-1) * d/dx(a) + a^b * ln(a) * d/dx(b)
        return Power.actual_derivative(base, base_dx, exponent, exponent_dx)

    @staticmethod
    def actual_derivative(base, base_dx, exponent, exponent_dx):
        # d/dx (a^b) = b * a^(b-1) * d/dx(a) + a^b * ln(a) * d/dx(b)
        return Add([
            Prod([
                Prod([
                    exponent,
                    Power([base, Add([exponent, Constant(-1)], 2)])
                ]),
                base_dx
            ]),
            Prod([
                Prod([
                    Power([base, exponent]),
                    Log([base])
                ]),
                exponent_dx]
            )
        ], 2)

    def clone(self) -> "Power":
        return Power([child.clone() for child in self.children])

    def to_python(self) -> str:
        return str(self.children[0]) + " ** (" + str(self.children[1]) + ")"
