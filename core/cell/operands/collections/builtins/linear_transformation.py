import numpy as np

from core.cell.optim.optimizable import OptimizableOperand
from core.cell.operands.function import PassThrough, Function
from core.cell.operands.weight import ShapedWeight


class LinearTransformation(OptimizableOperand):
    def __init__(self, dim_in, dim_out, optimizer=None):
        super().__init__(arity=1, optimizer=optimizer)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = ShapedWeight((dim_out, dim_in),
                                   np.random.randn(dim_out, dim_in))
        self.bias = ShapedWeight((dim_out,), np.zeros(dim_out))

    def __call__(self, x, meta_args=None):
        try:
            z = np.dot(self.weight.get(), x)
        except ValueError:
            z = np.dot(self.weight.get(), x[0])
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
            else:
                sw = ShapedWeight(
                    (self.dim_out, self.dim_out),
                    np.zeros((self.dim_out, self.dim_out))
                )
                sw.lock()
                return sw
        else:  # Derivative with respect to X
            return self._derive_x()

    def _derive_w(self):
        # The derivative of W * X + B with respect to W is X.
        def dW(args):
            X = np.array(args[0])
            # Repeat X to match the shape of W
            return X  # np.tile(X, (self.dim_out, 1))

        return Function(1, dW, [PassThrough(1)])

    def _derive_x(self):
        # The derivative of W * X + B with respect to X is W.
        def dX(args):
            return self.weight.get()

        return Function(1, dX, [PassThrough(1)])

    def _derive_b(self):
        # The derivative of W * X + B with respect to B is 1.
        def dB(args):
            return np.ones(self.dim_out)

        return Function(1, dB, [PassThrough(1)])

    def clone(self):
        cloned = LinearTransformation(self.dim_in, self.dim_out,
                                      self.optimizer.clone() if self.optimizer else None)
        cloned.weight = self.weight.clone()
        cloned.bias = self.bias.clone()
        return cloned

    def to_python(self) -> str:
        return f"{self.weight.to_python()} * x + {self.bias.to_python()}"

    def get_children(self):
        return [self.weight, self.bias]
