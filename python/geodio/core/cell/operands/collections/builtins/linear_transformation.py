import numpy as np

from geodio.core.cell.operands.function import Function
from geodio.core.cell.operands.constant import Constant
from geodio.core.cell.operands.weight import ShapedWeight
from geodio.core.cell.train import BOO
import geodio


def xavier_init(dim_out, dim_in):
    limit = np.sqrt(6. / (dim_in + dim_out))
    return np.random.uniform(-limit, limit, size=(dim_out, dim_in))


class LinearTransformation(BOO):
    def __init__(self, dim_in, dim_out, children, optimizer=None):
        super().__init__(children, optimizer=optimizer)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.in_data = None
        self.weight = ShapedWeight((dim_in, dim_out),
                                   np.random.randn(dim_in, dim_out))
        self.bias = ShapedWeight((dim_out,), np.zeros((dim_out,), float))
        self.dw = None
        self.db = None

    def forward(self, in_data, meta_args=None):
        self.in_data = in_data
        z = np.matmul(in_data, self.weight.get())
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
            self.db = - np.sum(dz, axis=0).reshape(-1, )
        else:
            self.db = - dz
        if np.ndim(self.in_data) == 1:
            self.in_data = self.in_data[np.newaxis, :]
        if np.ndim(dr) == 1:
            dr = dr[:, np.newaxis]
        self.dw = - np.matmul(self.in_data.T, dr)
        dx = np.matmul(dr, self.weight.get().T)
        dx = self.children[0].backpropagation(dx)
        return dx

    def get_local_gradients(self):
        return [self.dw, self.db]

    def get_operand_type(self):
        return geodio.geodio_bindings.OperandType.LinearTransformation

    def subscribe_to_graph(self, graph_wrapper, operand_type=None):
        if (hasattr(self, "graph_id") and self.graph_id is not None and
                self.graph_id != -1):
            return self.graph_id

        # Get a new ID for this operand
        self.graph_id = graph_wrapper.next_id()
        # Convert operand type to OperandType in C++
        operand_type = operand_type or self.get_operand_type()

        # Recursively add children to the graph
        x_id = self.children[0].subscribe_to_graph(graph_wrapper)
        w_id = self.weight.subscribe_to_graph(graph_wrapper)
        b_id = self.bias.subscribe_to_graph(graph_wrapper)
        child_ids = [x_id, w_id, b_id]

        # Add the operand to the C++ graph
        graph_wrapper.add_operand(operand_type, self.graph_id, child_ids)

        return self.graph_id
