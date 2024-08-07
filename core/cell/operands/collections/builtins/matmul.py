import numpy as np

from core.cell.operands.operand import Operand
from core.cell.operands.collections.builtins.add import Add
from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction


def matmul_any(a, b):
    # Get dimensions of a and b
    a_s_0, a_s_m1 = a.shape[0], a.shape[-1]
    b_s_0, b_s_m1 = b.shape[0], b.shape[-1]

    # Case 1: Perfect match for matrix multiplication
    if a.shape == (1, 1) and b_s_0 != 1:
        r = a[0] * b
    elif a_s_m1 == b_s_0 and np.ndim(b) >= 2:
        if b_s_m1 != 1:
            r = a @ b
        else:
            r = b * a
    # Case 2: Handle when b is a vector and needs reshaping
    elif np.ndim(b) == 1 and a_s_m1 == b_s_0:
        b = b[:, np.newaxis]
        if a_s_m1 != a_s_0:
            r = a @ b
        else:
            r = b * a
    # Case 3: Handle broadcasting and element-wise multiplication
    elif a_s_0 == b_s_0:
        if np.ndim(b) == 1:
            b = b[:, np.newaxis]
        b_s_m1 = b.shape[-1]
        if a_s_m1 == b_s_m1:
            r = a @ b.T
        elif a_s_m1 == 1 or b_s_m1 == 1:
            r = b * a
        else:
            r = b @ a
    # Case 4: Handle cases where a is a column vector and b is a vector
    elif a_s_m1 == 1 and np.ndim(b) == 1:
        b = np.atleast_2d(b)
        r = a @ b
    # Default case: Use tensordot for other scenarios
    else:
        if a_s_0 == b_s_m1:
            r = b @ a
        else:
            r = np.tensordot(a.T, b, axes=0)

    return r


def matmul_any_original(a, b):
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
    elif a_s_m1 == 1 and np.ndim(b) == 1:
        b = np.atleast_2d(b)
        r = a @ b
    else:
        r = np.tensordot(a.T, b, axes=0)
    return r


class Matmul(BuiltinBaseFunction):
    def __init__(self, children, flag_original=False):
        super().__init__(children, "matmul", 2)
        self.flag_original = flag_original

    def __call__(self, args, meta_args=None):
        a = self.children[0](args, meta_args)
        b = self.children[1](args, meta_args)
        if np.isscalar(a) and np.isscalar(b):
            return a * b
        if not self.flag_original:
            r = matmul_any(a, b)
        else:
            r = matmul_any_original(a, b)
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


def matmul_of(operand_a: Operand, operand_b: Operand,
              flag_original=False) -> Matmul:
    if isinstance(operand_b, Matmul):
        child_a = operand_b.children[0]
        child_b = operand_b.children[1]
        result = matmul_of(operand_a, child_a)
        result = matmul_of(result, child_b)
    else:
        result = Matmul([operand_a, operand_b], flag_original)
    return result
