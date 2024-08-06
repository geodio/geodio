import numpy as np

from core.cell.operands.operand import Operand
from core.cell.operands.collections.builtins.add import Add
from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction


def matmul_any(a, b):
    b_s_0 = b.shape[0]
    a_s_m1 = a.shape[-1]
    a_s_0 = a.shape[0]
    if a.shape == (1, 1) and b_s_0 != 1:
        r = a[0] * b
    elif a_s_m1 == b_s_0 and np.ndim(b) >= 2:
        r = a @ b
    elif a_s_0 == b_s_0:
        if np.ndim(b) == 1:
            b = b[:, np.newaxis]
        b_s_m1 = b.shape[-1]
        if b_s_m1 == 1 or a_s_m1 == b_s_m1 or a_s_m1 == 1:
            r = b * a
        else:
            r = b @ a
    elif np.ndim(b) == 1 and a_s_m1 == b_s_0:
        b = b[:, np.newaxis]
        r = a @ b
    elif a_s_m1 == 1 and np.ndim(b) == 1:
        b = np.atleast_2d(b)
        r = a @ b
    else:
        r = np.tensordot(a.T, b, axes=0)
    return r


class Matmul(BuiltinBaseFunction):
    def __init__(self, children):
        super().__init__(children, "matmul", 2)

    def __call__(self, args, meta_args=None):
        a = self.children[0](args, meta_args)
        b = self.children[1](args, meta_args)
        if np.isscalar(a) and np.isscalar(b):
            return a * b
        r = matmul_any(a, b)
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


def matmul_of(operand_a: Operand, operand_b: Operand) -> Matmul:
    if isinstance(operand_b, Matmul):
        child_a = operand_b.children[0]
        child_b = operand_b.children[1]
        result = matmul_of(operand_a, child_a)
        result = matmul_of(result, child_b)
    else:
        result = Matmul([operand_a, operand_b])
    return result
