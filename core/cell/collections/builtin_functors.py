import sys
import time

import numpy as np

from core.cell.collections.functors import Functor, CollectionBasedFunctors
from core.cell.operands.constant import Constant
from core.cell.operands.operand import Operand
from core.cell.optim.optimizable import OptimizableOperand
from core.cell.optim.optimization_args import OptimizationArgs


def clean_number(x):
    if isinstance(x, np.ndarray):
        return x
    try:
        if x is None or str(x) == 'nan' or str(x) == 'inf' or np.isinf(x):
            return 0
    except Exception:
        return 0
    return x


class BuiltinFunctor(Functor):
    def __init__(self, children, func_id, arity):
        self.__name__ = func_id
        super().__init__(func_id, None, arity)
        self.children = children
        self.value = self


class Add(BuiltinFunctor):
    def __init__(self, children, arity):
        super().__init__(children, f"add_{arity}", arity)

    def __call__(self, x):
        return clean_number(sum([child(x) for child in self.children]))

    def derive(self, index, by_weights=True):
        return Add(
            [child.derive(index, by_weights) for child in self.children],
            arity=self.arity)

    def clone(self) -> "Add":
        return Add([child.clone() for child in self.children], self.arity)

    def to_python(self) -> str:
        return " + ".join(child.__repr__() for child in self.children)


class Prod(BuiltinFunctor):
    def __init__(self, children):
        super().__init__(children, "prod", 2)

    def __call__(self, x):
        a = self.children[0](x)
        b = self.children[1](x)
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


class Dot(BuiltinFunctor):
    def __init__(self, children):
        super().__init__(children, "dot_prod", 2)

    def __call__(self, x):
        a = self.children[0](x)
        b = self.children[1](x)

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

    def clone(self) -> "Prod":
        return Prod([child.clone() for child in self.children])


class Max(BuiltinFunctor):
    def __init__(self, children, arity):
        super().__init__(children, f"max_{arity}", arity)

    def __call__(self, x):
        return max([child(x) for child in self.children])

    def d(self, dx):
        # TODO the derivative is not correctly computed
        return Max([child.d(dx) for child in self.children], arity=self.arity)

    def clone(self) -> "Max":
        return Max([child.clone() for child in self.children], self.arity)


class Power(BuiltinFunctor):
    def __init__(self, children):
        super().__init__(children, "power", 2)

    def __call__(self, x):
        base_func = self.children[0]
        exponent = self.children[1](x)
        try:
            if exponent == 0:
                return 1.0
            return clean_number(np.power(0.0 + base_func(x), exponent))
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


class Sub(BuiltinFunctor):
    def __init__(self, children):
        super().__init__(children, "sub", 2)

    def __call__(self, x):
        try:
            return clean_number(self.children[0](x) - self.children[1](x))
        except IndexError:
            return 0

    def derive(self, index, by_weights=True):
        return Sub([self.children[0].derive(index, by_weights),
                    self.children[1].derive(index, by_weights)])

    def clone(self) -> "Sub":
        return Sub([child.clone() for child in self.children])

    def to_python(self) -> str:
        return str(self.children[0]) + " - " + str(self.children[1])


class Div(BuiltinFunctor):
    def __init__(self, children):
        super().__init__(children, "div", 2)

    def __call__(self, x):
        try:
            up = self.children[0](x)
            down = self.children[1](x)
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
            Power([b, Constant(2)])
        ])

    def clone(self) -> "Div":
        return Div([child.clone() for child in self.children])

    def to_python(self) -> str:
        return ("(" + str(self.children[0]) + ") / (" + str(self.children[1])
                + ")")


class Log(BuiltinFunctor):
    def __init__(self, children):
        super().__init__(children, "log", 1)

    def __call__(self, x):
        try:
            return clean_number(np.log(self.children[0](x)))
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


BUILTIN_FUNCTORS = CollectionBasedFunctors()
BUILTIN_FUNCTORS.add_functor(Add([], 2))
BUILTIN_FUNCTORS.add_functor(Add([], 3))
BUILTIN_FUNCTORS.add_functor(Add([], 4))
BUILTIN_FUNCTORS.add_functor(Add([], 5))
BUILTIN_FUNCTORS.add_functor(Prod([]))
BUILTIN_FUNCTORS.add_functor(Log([]))
BUILTIN_FUNCTORS.add_functor(Div([]))
BUILTIN_FUNCTORS.add_functor(Sub([]))
BUILTIN_FUNCTORS.add_functor(Power([]))

class T(OptimizableOperand):
    def __init__(self, arity, x: Operand):
        super().__init__(arity)
        self.x = x

    def __call__(self, x):
        return self.x(x).T

    def derive(self, index, by_weight=True):
        pass

    def clone(self):
        return T(self.arity, self.x.clone())

    def to_python(self) -> str:
        return self.x.to_python() + ".T"

    def get_sub_items(self):
        return [self.x]

    def optimize(self, args: OptimizationArgs):
        pass


class Linker(OptimizableOperand):

    def __init__(self, arity, f: Operand, g: Operand):
        super().__init__(arity)
        self.f = f
        self.g = g

    def __call__(self, x):
        return self.f(self.g(x))

    def derive(self, index, by_weight=True):
        """
        (f(g(x)))' = f'(g(x)) * g'(x)
        :param index:
        :param by_weight:
        :return:
        """
        chain = Linker(self.arity, self.f.derive(index, by_weight), self.g)
        chained = T(1, self.g.derive(index, by_weight))
        derivative = Prod([chain, chained])
        return derivative

    def clone(self):
        return Linker(self.arity, self.f.clone(), self.g.clone())

    def to_python(self) -> str:
        return self.f.to_python() + "(" + self.g.to_python() + ")"

    def get_sub_items(self):
        return [self.f, self.g]

    def optimize(self, args: OptimizationArgs):
        pass
