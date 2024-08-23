from core.cell.operands.collections.basefunctions import \
    CollectionBasedBaseFunctions
from core.cell.operands.collections.builtins.add import *
from core.cell.operands.collections.builtins.div import *
from core.cell.operands.collections.builtins.linker import *
from core.cell.operands.collections.builtins.log import *
from core.cell.operands.collections.builtins.matmul import *
from core.cell.operands.collections.builtins.max import *
from core.cell.operands.collections.builtins.power import *
from core.cell.operands.collections.builtins.prod import *
from core.cell.operands.collections.builtins.sub import *
from core.cell.operands.collections.builtins.transpose import *
from core.cell.operands.collections.builtins.dot import *
from core.cell.operands.collections.builtins.linear_transformation import *
from core.cell.operands.collections.builtins.seq import *
from core.cell.operands.operand import *


def add(o1: Operand, o2: Operand):
    return Add([o1, o2], 2)


def power(o1: Operand, o2: Operand):
    if not isinstance(o2, Operand):
        o2 = Constant(o2)
    return Power([o1, o2])


def div(o1: Operand, o2: Operand):
    return Div([o1, o2])


def link(o1: Operand, o2: Operand):
    return Linker(o2, o1)


def sub(o1: Operand, o2: Operand):
    return Sub([o1, o2])


BUILTIN_FUNCTORS = CollectionBasedBaseFunctions()
BUILTIN_FUNCTORS.add_functor(Add([], 2))
BUILTIN_FUNCTORS.add_functor(Add([], 3))
BUILTIN_FUNCTORS.add_functor(Add([], 4))
BUILTIN_FUNCTORS.add_functor(Add([], 5))
BUILTIN_FUNCTORS.add_functor(Prod([]))
BUILTIN_FUNCTORS.add_functor(Log([]))
BUILTIN_FUNCTORS.add_functor(Div([]))
BUILTIN_FUNCTORS.add_functor(Sub([]))
BUILTIN_FUNCTORS.add_functor(Power([]))

GLOBAL_BUILTINS["matmul"] = matmul_of
GLOBAL_BUILTINS["transpose"] = transpose_of
GLOBAL_BUILTINS["add"] = add
GLOBAL_BUILTINS["power"] = power
GLOBAL_BUILTINS["div"] = div
GLOBAL_BUILTINS["link"] = link
GLOBAL_BUILTINS["sub"] = sub
