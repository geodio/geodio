import sys

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

    def get_output_dim(self):
        # TODO consider tuple output
        max_dim = 0
        for child in self.children:
            output_dim = child.get_output_dim()
            if isinstance(output_dim, int):
                max_dim = max(max_dim, output_dim)
            elif isinstance(output_dim, tuple):
                max_dim = max(max_dim, len(output_dim))
        return max_dim


class Add(BuiltinFunctor):
    def __init__(self, children, arity):
        super().__init__(children, f"add_{arity}", arity)

    def __call__(self, args, meta_args=None):
        return clean_number(sum([child(args, meta_args) for child in self.children]))

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


class Dot(BuiltinFunctor):
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


class Matmul(BuiltinFunctor):
    def __init__(self, children):
        super().__init__(children, "matmul", 2)

    def __call__(self, args, meta_args=None):
        a = self.children[0](args, meta_args)
        b = self.children[1](args, meta_args)
        if np.ndim(b) == 1:
            b = np.array([b])
        try:
            r = a @ b
        except ValueError:
            r = b * a
        return r

    def derive(self, index, by_weights=True):
        return Add(
            [
                Matmul([self.children[0].derive(index, by_weights),
                        self.children[1]]),
                Matmul([self.children[0], self.children[1].derive(index,
                                                                  by_weights)])
            ],
            2
        )

    def clone(self) -> "Matmul":
        return Matmul([child.clone() for child in self.children])


class Max(BuiltinFunctor):
    def __init__(self, children, arity):
        super().__init__(children, f"max_{arity}", arity)

    def __call__(self, args, meta_args=None):
        return max([child(args, meta_args) for child in self.children])

    def d(self, dx):
        # TODO the derivative is not correctly computed
        return Max([child.d(dx) for child in self.children], arity=self.arity)

    def clone(self) -> "Max":
        return Max([child.clone() for child in self.children], self.arity)


class Power(BuiltinFunctor):
    def __init__(self, children):
        super().__init__(children, "power", 2)

    def __call__(self, args, meta_args=None):
        base_func = self.children[0]
        exponent = self.children[1](args, meta_args)
        try:
            if exponent == 0:
                return 1.0
            return clean_number(np.power(0.0 + base_func(args, meta_args), exponent))
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

    def __call__(self, args, meta_args=None):
        try:
            return clean_number(self.children[0](args, meta_args) - self.children[1](args, meta_args))
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


class Transpose(OptimizableOperand):
    def __init__(self, arity, x: Operand):
        super().__init__(arity)
        self.x = x

    def __call__(self, args, meta_args=None):
        out = self.x(args, meta_args)
        if np.ndim(out) == 1:
            out = out[:, np.newaxis]
            return out
        r = out.T

        return r

    def derive_unchained(self, index, by_weight=True):
        pass

    def clone(self):
        return Transpose(self.arity, self.x.clone())

    def to_python(self) -> str:
        return self.x.to_python() + ".T"

    def get_sub_items(self):
        return [self.x]


def matmul_of(operand_a: Operand, operand_b: Operand) -> Matmul:
    if isinstance(operand_b, Matmul):
        child_a = operand_b.children[0]
        child_b = operand_b.children[1]
        result = matmul_of(operand_a, child_a)
        result = matmul_of(result, child_b)
    else:
        result = Matmul([operand_a, operand_b])
    return result


def transpose_of(operand: Operand) -> Transpose:
    if isinstance(operand, Matmul):
        child_a = operand.children[0]
        child_b = operand.children[1]
        result = matmul_of(transpose_of(child_b), transpose_of(child_a))
    elif isinstance(operand, Transpose):
        result = operand.x
    else:
        result = Transpose(1, operand)
    return result


class Linker(OptimizableOperand):

    def __init__(self, f: Operand, g: Operand, input_shape=0,
                 mark=False):
        super().__init__(g.arity)
        self.f = f
        self.g = g
        self.input_shape = input_shape
        self.mark = mark

    def __call__(self, args, meta_args=None):
        x_ = [self.g(args, meta_args)]
        return self.f(x_, meta_args)

    def derive_unchained(self, index, by_weight=True):
        """
        (f(g(x)))' = f'(g(x)) * g'(x)
        :param index:
        :param by_weight:
        :return:
        """
        if by_weight and self.g.is_independent_of(index):
            derivative = self.__derive_chained_f(index)
        else:
            derivative = self.__derive_unchained_g(by_weight, index)
        return derivative

    def __derive_chained_f(self, index):
        self_double = Linker(self.f.derive(index, True), self.g)
        return self_double

    def __derive_unchained_g(self, by_weight, index):
        chain = Linker(self.f.derive(0, False), self.g)
        chained = self.g.derive(index, by_weight)
        derivative = matmul_of(transpose_of(chain), chained)
        return derivative

    def clone(self):
        return Linker(self.f.clone(), self.g.clone())

    def to_python(self) -> str:
        return "[λX.[" + self.f.to_python() + "]" + self.g.to_python() + "]"

    def get_sub_items(self):
        return [self.g, self.f]
