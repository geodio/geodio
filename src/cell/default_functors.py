import sys

import numpy as np

from src.cell.functors import Functor, CollectionBasedFunctors
from src.cell.operands.constant import Constant


class DefaultFunctor(Functor):
    def __init__(self, children, func_id, arity):
        self.children = children
        self.__name__ = func_id
        super().__init__(func_id, None, arity)
        self.value = self


class Add(DefaultFunctor):
    def __init__(self, children, arity):
        super().__init__(children, f"add_{arity}", arity)

    def __call__(self, x):
        return sum([child(x) for child in self.children])

    def d(self, dx):
        return Add([child.d(dx) for child in self.children], arity=self.arity)

    def clone(self) -> "Add":
        return Add([child.clone() for child in self.children], self.arity)


class Prod(DefaultFunctor):
    def __init__(self, children):
        super().__init__(children, "prod", 2)

    def __call__(self, x):
        return self.children[0](x) * self.children[1](x)

    def d(self, dx):
        return Add(
            [
                Prod([self.children[0](dx), self.children[1]]),
                Prod([self.children[0], self.children[1](dx)])
            ],
            2
        )

    def clone(self) -> "Prod":
        return Prod([child.clone() for child in self.children])


class Max(DefaultFunctor):
    def __init__(self, children, arity):
        super().__init__(children, f"max_{arity}", arity)

    def __call__(self, x):
        return max([child(x) for child in self.children])

    def d(self, dx):
        # TODO the derivative is not correctly computed
        return Max([child.d(dx) for child in self.children], arity=self.arity)

    def clone(self) -> "Max":
        return Max([child.clone() for child in self.children], self.arity)


class Power(DefaultFunctor):
    def __init__(self, children):
        super().__init__(children, "power", 2)

    def __call__(self, x):
        base_func = self.children[0]
        exponent = self.children[1](x)
        if exponent == 0:
            return 1.0
        return np.power(0.0 + base_func(x), exponent)

    def d(self, dx):
        base = self.children[0]
        exponent = self.children[1]
        base_dx = base.d(dx)
        exponent_dx = exponent.d(dx)

        # d/dx (a^b) = b * a^(b-1) * d/dx(a) + a^b * ln(a) * d/dx(b)
        return Add([
            Prod([
                exponent,
                Power([base, Add([exponent, Constant(-1)], 2)]),
                base_dx
            ]),
            Prod([
                Power([base, exponent]),
                Log([base]),
                exponent_dx
            ])
        ], 2)

    def clone(self) -> "Power":
        return Power([child.clone() for child in self.children])


class Sub(DefaultFunctor):
    def __init__(self, children):
        super().__init__(children, "sub", 2)

    def __call__(self, x):
        return self.children[0](x) - self.children[1](x)

    def d(self, dx):
        return Sub([self.children[0].d(dx), self.children[1].d(dx)])

    def clone(self) -> "Sub":
        return Sub([child.clone() for child in self.children])


class Div(DefaultFunctor):
    def __init__(self, children):
        super().__init__(children, "div", 2)

    def __call__(self, x):
        up = self.children[0](x)
        down = self.children[1](x)
        if down == 0:
            if up >= 0:
                return sys.maxsize
            return -sys.maxsize
        return up / down

    def d(self, dx):
        # d/dx (a / b) = (b * d/dx(a) - a * d/dx(b)) / (b^2)
        a, b = self.children[0], self.children[1]
        return Div([
            Sub([
                Prod([b, a.d(dx)]),
                Prod([a, b.d(dx)])
            ]),
            Power([b, 2])
        ])

    def clone(self) -> "Div":
        return Div([child.clone() for child in self.children])


class Log(DefaultFunctor):
    def __init__(self, children):
        super().__init__(children, "log", 1)

    def __call__(self, x):
        try:
            return np.log(self.children[0](x))
        except:
            return 0

    def d(self, dx):
        # d/dx (log(a)) = 1 / a * d/dx(a)
        a = self.children[0]
        return Div([
            a.d(dx),
            a
        ])

    def clone(self) -> "Log":
        return Log([child.clone() for child in self.children])


DEFAULT_FUNCTORS = CollectionBasedFunctors()
DEFAULT_FUNCTORS.add_functor(Add([], 2))
DEFAULT_FUNCTORS.add_functor(Add([], 3))
DEFAULT_FUNCTORS.add_functor(Add([], 4))
DEFAULT_FUNCTORS.add_functor(Add([], 5))
DEFAULT_FUNCTORS.add_functor(Prod([]))
DEFAULT_FUNCTORS.add_functor(Log([]))
DEFAULT_FUNCTORS.add_functor(Div([]))
DEFAULT_FUNCTORS.add_functor(Sub([]))
DEFAULT_FUNCTORS.add_functor(Power([]))
