import numpy as np

from core.cell.train.optimizable import OptimizableOperand, Operand
from core.cell.operands.collections.builtins.matmul import Matmul, matmul_of


class Transpose(OptimizableOperand):
    def __init__(self, arity, x: Operand):
        super().__init__(arity)
        self.x = x

    def __call__(self, args, meta_args=None):
        out = self.x(args, meta_args)
        if np.isscalar(out):
            return out
        if np.ndim(out) == 1:
            out = out[:, np.newaxis]
            return out
        r = out.T

        return r

    def derive_uncached(self, index, by_weight=True):
        pass

    def clone(self):
        return Transpose(self.arity, self.x.clone())

    def to_python(self) -> str:
        return self.x.to_python() + ".T"

    def get_children(self):
        return [self.x]


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
