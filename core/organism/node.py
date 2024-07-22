import sys
from typing import Union, List

import numpy as np

from core.cell.collections.builtin_functors import Prod, Add, Dot, Sub, Linker
from core.cell.operands.constant import ONE
from core.cell.operands.function import Function, PassThrough
from core.cell.operands.operand import Operand
from core.cell.operands.variable import Variable
from core.cell.operands.weight import AbsWeight, t_weight
from core.cell.optim.loss import MSE, LossFunction
from core.cell.optim.optimizable import OptimizableOperand
from core.cell.optim.optimization_args import OptimizationArgs
from core.cell.optim.optimizer import Optimizer


class ShapedWeight(AbsWeight):
    def __init__(self, shape, weight: Union[np.ndarray, float] = None):
        super().__init__(adaptive_shape=False)
        self.shape = shape
        if weight is None:
            self.__weight = np.zeros(shape)
        else:
            self.set(weight)

    def set(self, weight: Union[np.ndarray, float, 't_weight']):
        if isinstance(weight, AbsWeight):
            weight = weight.get()
        if isinstance(weight, np.ndarray):
            if weight.shape != self.shape:
                raise ValueError(f"Weight shape {weight.shape} does not "
                                 f"match required shape {self.shape}.")
            self.__weight = weight
        else:
            self.__weight = np.full(self.shape, weight)

    def get(self) -> np.ndarray:
        return self.__weight

    def d(self, var_index):
        derivative = ShapedWeight(self.shape, np.zeros(self.shape))
        derivative.lock()
        return derivative

    def d_w(self, dw):
        if self.w_index == dw:
            derivative = ShapedWeight(self.shape, np.ones(self.shape))
        else:
            derivative = ShapedWeight(self.shape, np.zeros(self.shape))
        derivative.lock()
        return derivative

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def clone(self) -> 'ShapedWeight':
        sw = ShapedWeight(self.shape, np.copy(self.__weight))
        sw.w_index = self.w_index
        return sw

    def to_python(self) -> str:
        return str(self.__weight)


class Node(OptimizableOperand):
    def __init__(self, arity, dim_in, dim_out, activ_fun, optimizer=None,
                 nxt=None):
        super().__init__(arity, optimizer)
        self.arity = arity
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activ_fun = activ_fun
        self.optimizer = optimizer
        self.nxt = nxt

        self.weight = ShapedWeight(
            (dim_out, dim_in), np.random.randn(dim_out, dim_in)
        )
        self.bias = ShapedWeight(
            (dim_out,), np.zeros(dim_out)
        )

        self.input_data = None
        self.z = None
        self.activated_output = None
        self.derivative_cache = {}

    def __call__(self, args):
        try:
            self.input_data = np.array(args[0])
            self.z = np.dot(self.weight.get(),
                            self.input_data) + self.bias.get()
            self.activated_output = self.activ_fun(self.z)
            if self.nxt:
                to_be_returned = self.nxt([self.activated_output])
                return to_be_returned
            return self.activated_output
        except ValueError:
            return self.activated_output

    def optimize(self, args: OptimizationArgs):
        self.optimizer(self, args)

    def get_sub_items(self):
        # return [self.weight, self.bias, self.activ_fun]
        if self.nxt is None:
            return [self.weight, self.bias]
        else:
            return [self.weight, self.nxt, self.bias]

    def to_python(self) -> str:
        return self.activ_fun.to_python() + "{\n" + (
                "\ty * " + self.weight.to_python() + " + " +
                self.bias.to_python()
        ) + "\n}"

    def derive(self, index, by_weights=True):
        derivative_id = f'{"W" if by_weights else "X"}_{index}'
        if derivative_id not in self.derivative_cache:
            if self.nxt is not None:
                derivative = self.derive_chained(index, by_weights)
            else:
                derivative = self.derive_unchained(index, by_weights)
            self.derivative_cache[derivative_id] = derivative
        return self.derivative_cache[derivative_id]

    def derive_chained(self, index, by_weights=True):
        self.get_weights()
        clone = self.clone()
        clone.nxt = None
        clone.weight = self.weight
        clone.bias = self.bias
        chained = Linker(1, self.nxt, clone)
        return chained

    def derive_unchained(self, index, by_weights=True):
        activ_fun_derived = self.activ_fun.derive(index, by_weights)
        clone = self.clone()
        clone.nxt = None
        clone.activ_fun = activ_fun_derived
        clone.weight = self.weight
        clone.bias = self.bias

        function = Dot([self.weight, Variable(0)])
        z_function = Add(
            [function, self.bias], 1
        )
        link = Linker(1, self.activ_fun, z_function)

        unchained = link.derive(index, by_weights)
        return unchained

    def clone(self):
        cloned = Node(self.arity, self.dim_in, self.dim_out,
                      self.activ_fun.clone(),
                      self.optimizer.clone() if self.optimizer else None,
                      self.nxt.clone() if self.nxt else None)
        cloned.weight = self.weight.clone()
        cloned.bias = self.bias.clone()
        if self.nxt:
            cloned.nxt.weight = self.nxt.weight
            cloned.nxt.bias = self.nxt.bias
        return cloned


class BackpropagationOptimizer(Optimizer):
    def __init__(self):
        super().__init__()
        self.weight_grad = None
        self.bias_grad = None

    def optimize(self, node: Node, args: OptimizationArgs):
        self._compute_gradients(node, args)
        self._update_weights(node, args.learning_rate)

    def _compute_gradients(self, node: Node, args: OptimizationArgs):
        if node.nxt and node.nxt.optimizer.weight_grad:
            next_grad = np.dot(
                node.nxt.weight.get().T, node.nxt.optimizer.weight_grad
            )
        else:
            next_grad = args.loss_function.gradient(
                node, node.activated_output, args.desired_output, 1
            )

        local_grad = node.activ_fun.derive(node.z)
        delta = next_grad * local_grad

        self.weight_grad = np.outer(delta, node.input_data)
        self.bias_grad = delta

    def _update_weights(self, node: Node, learning_rate: float):
        node.weight.set(node.weight.get() - learning_rate * self.weight_grad)
        node.bias.set(node.bias.get() - learning_rate * self.bias_grad)

    def clone(self):
        return BackpropagationOptimizer()


# Example usage
class SimpleActivation(OptimizableOperand):
    def __init__(self, arity):
        super().__init__(arity)

    def __call__(self, x):
        return np.maximum(0, x)  # ReLU activation

    def clone(self):
        return SimpleActivation(self.arity)

    def derive(self, index, by_weights=True):
        # Derivative of ReLU
        return lambda x: np.where(x > 0, 1, 0)

    def optimize(self, args: OptimizationArgs):
        pass

    def to_python(self) -> str:
        return "SimpleActivation()"


class SigmoidActivation(Operand):
    def __init__(self, arity):
        super().__init__(arity)

        def d_sigmoid(z):
            x = 1 / (1 + np.exp(-z))
            return x * (1 - x)

        self._derivative = Function(1, d_sigmoid, [PassThrough(1)])

    def __call__(self, args):
        return 1 / (1 + np.exp(-args))

    def __invert__(self):
        pass

    def clone(self) -> "Operand":
        return SigmoidActivation(self.arity)

    def to_python(self) -> str:
        return "sigmoid"

    def derive(self, index, by_weights=True):
        return self._derivative


def main():
    dim_in = 5
    dim_out = 3
    arity = 1

    activation_function = SigmoidActivation(arity)

    node2 = Node(arity, dim_out, dim_out, activation_function,
                 optimizer=Optimizer())
    node1 = Node(arity, dim_in, dim_out, activation_function,
                 optimizer=Optimizer(), nxt=node2)
    node1.set_optimization_risk(True)
    input_data = [
        [np.array([1, 1, 1, 1, 1])],
        [np.array([0, 1, 1, 0, 1])],
        [np.array([1, 1, 0, 1, 1])],
        [np.array([1, 0, 0, 1, 0])]
    ]
    desired_output = [
        [np.array([1.0, 1.0, 1.0])],
        [np.array([0.0, 1.0, 0.0])],
        [np.array([1.0, 0.0, 1.0])],
        [np.array([0.0, 0.0, 0.0])]
    ]

    optimization_args = OptimizationArgs(
        inputs=input_data,
        desired_output=desired_output,
        loss_function=MSE(),
        learning_rate=0.1,
        max_iter=10000,
        min_error=sys.maxsize
    )
    output_before = [node1(input_data_i) for input_data_i in input_data]
    print("Weights before optimization:")
    print([[w, w.w_index] for w in node1.get_weights()])
    print(MSE().evaluate(node1, input_data, desired_output))
    node1.optimize(optimization_args)
    print("Weights after optimization:")
    print([[w, w.w_index] for w in node1.get_weights()])
    print(MSE().evaluate(node1, input_data, desired_output))
    output = [node1(input_data_i) for input_data_i in input_data]

    print("            Input         :", input_data)
    print("Before Optimization output:", output_before)
    print("       Final output       :", output)
    print("       Desired output     :", desired_output)


if __name__ == "__main__":
    main()
