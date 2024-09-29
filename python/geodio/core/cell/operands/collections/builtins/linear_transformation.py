import numpy as np

from geodio.core.cell.operands.function import Function
from geodio.core.cell.operands.constant import Constant
from geodio.core.cell.operands.weight import ShapedWeight
from geodio.core.cell.train import BOO


def xavier_init(dim_out, dim_in):
    limit = np.sqrt(6. / (dim_in + dim_out))
    return np.random.uniform(-limit, limit, size=(dim_out, dim_in))


class LinearTransformation(BOO):
    def __init__(self, dim_in, dim_out, children, optimizer=None):
        super().__init__(children, optimizer=optimizer)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.in_data = None
        self.weight = ShapedWeight((dim_out, dim_in),
                                   np.random.randn(dim_out, dim_in))
        self.bias = ShapedWeight((dim_out,), np.zeros(dim_out, ))
        self.dw = None
        self.db = None

    def forward(self, in_data, meta_args=None):
        self.in_data = in_data
        z = np.matmul(self.weight.get(), in_data)
        try:
            r = z + self.bias.get()
        except ValueError:
            r = z + self.bias.get()[:, np.newaxis]
        return r

    def derive_uncached(self, index, by_weights=True):
        if by_weights:
            if index == self.weight.w_index:  # Derivative with respect to W
                return self._derive_w()
            elif index == self.bias.w_index:  # Derivative with respect to B
                return self._derive_b()

        clone = self.clone()
        clone.children[0] = self.children[0].derive(index, by_weights)
        clone.bias.set(np.zeros(self.dim_out))
        return clone

    def _derive_w(self):

        func = lambda x: np.tile(x, (self.dim_out, 1))
        derivative = Function(1, func, self.children)
        return derivative

    def _derive_b(self):
        # The derivative of W * X + B with respect to B is 1.
        sw = Constant(
            np.ones((self.dim_out,))
        )
        return sw

    def clone(self) -> "LinearTransformation":
        cloned = LinearTransformation(self.dim_in, self.dim_out,
                                      [child.clone() for child in
                                       self.children],
                                      self.optimizer.clone()
                                      if self.optimizer else None)
        cloned.weight = self.weight.clone()
        cloned.bias = self.bias.clone()
        return cloned

    def to_python(self) -> str:
        return f"{self.weight.to_python()} * x + {self.bias.to_python()}"

    def get_sub_operands(self):
        return [self.weight, self.bias, self.children[0]]

    def backpropagation(self, dx: np.ndarray, meta_args=None) -> np.ndarray:
        dz = dx
        dr = dz.copy()
        if np.ndim(dz) == 2:
            self.db = - np.sum(dz, axis=1).reshape(-1, )
        else:
            self.db = - dz
        self.dw = - np.matmul(dr, self.in_data.T)
        dx = np.matmul(self.weight.get().T, dr)
        dx = self.children[0].backpropagation(dx)
        return dx

    def get_local_gradients(self):
        return [self.dw, self.db]

    def get_operand_type(self):
        return geodio.geodio_bindings.OperandType.LinearTransformation
