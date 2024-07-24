import sys
from typing import Union

import numpy as np

from core.cell.collections.builtin_functors import Add, Dot, Linker
from core.cell.operands.function import Function, PassThrough
from core.cell.operands.variable import Variable, AdaptiveConstant
from core.cell.operands.weight import AbsWeight, t_weight
from core.cell.optim.loss import MSEMultivariate
from core.cell.optim.optimizable import OptimizableOperand
from core.cell.optim.optimization_args import OptimizationArgs
from core.cell.optim.optimizer import Optimizer
from core.organism.activation_function import SigmoidActivation, \
    ActivationFunction


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


class LinearTransformation(OptimizableOperand):
    def __init__(self, dim_in, dim_out, optimizer=None):
        super().__init__(arity=1, optimizer=optimizer)
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.weight = ShapedWeight((dim_out, dim_in),
                                   np.random.randn(dim_out, dim_in))
        self.bias = ShapedWeight((dim_out,), np.zeros(dim_out))

    def __call__(self, args):
        X = np.array(args[0])
        return np.dot(self.weight.get(), X) + self.bias.get()

    def derive(self, index, by_weights=True):
        if by_weights:
            if index == self.weight.w_index:  # Derivative with respect to W
                return self._derive_w()
            elif index == self.bias.w_index:  # Derivative with respect to B
                return self._derive_b()
            else:
                return ShapedWeight((self.dim_out,), np.zeros(self.dim_out))
        else:  # Derivative with respect to X
            return self._derive_x()

    def _derive_w(self):
        # The derivative of W * X + B with respect to W is X.
        def dW(args):
            X = np.array(args[0])
            # Repeat X to match the shape of W
            return np.tile(X, (self.dim_out, 1))

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
        return f"np.dot({self.weight.to_python()}, X) + {self.bias.to_python()}"

    def get_sub_items(self):
        return [self.weight, self.bias]

    def optimize(self, args):
        self.optimizer(self, args)


class Node(OptimizableOperand):
    def __init__(self, arity, dim_in, dim_out, activ_fun: ActivationFunction,
                 optimizer=None):
        super().__init__(arity, optimizer)
        self.arity = arity
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.activ_fun = activ_fun
        self.optimizer = optimizer

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
        self.output_dimensionality = dim_out

    def __call__(self, args):
        try:
            self.input_data = np.array(args[0])
            self.z = np.dot(self.weight.get(),
                            self.input_data) + self.bias.get()
            self.activated_output = self.activ_fun([self.z])
            return self.activated_output
        except ValueError:
            return self.activated_output

    def optimize(self, args: OptimizationArgs):
        self.optimizer(self, args)

    def get_sub_items(self):
        return [self.weight, self.bias]

    def to_python(self) -> str:
        return self.activ_fun.to_python() + "{\n" + (
                "\ty * " + self.weight.to_python() + " + " +
                self.bias.to_python()
        ) + "\n}"

    def derive(self, index, by_weights=True):
        derivative_id = f'{"W" if by_weights else "X"}_{index}'
        if derivative_id not in self.derivative_cache:
            derivative = self.derive_unchained(index, by_weights)
            self.derivative_cache[derivative_id] = derivative
        return self.derivative_cache[derivative_id]

    def derive_unchained(self, index, by_weights=True):
        z_function = LinearTransformation(self.dim_in, self.dim_out)
        z_function.weight = self.weight
        z_function.bias = self.bias
        link = Linker(1, self.activ_fun, z_function, (self.dim_in,))
        unchained = link.derive(index, by_weights)
        print("Unchained Derivative Shape:",
              unchained([np.zeros(self.dim_in)]).shape)
        return unchained

    def clone(self):
        cloned = Node(self.arity, self.dim_in, self.dim_out,
                      self.activ_fun.clone(),
                      self.optimizer.clone() if self.optimizer else None)
        cloned.weight = self.weight.clone()
        cloned.bias = self.bias.clone()
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


def main():
    dim_in = 5
    dim_out = 3
    arity = 1

    activation_function = SigmoidActivation()

    node2 = Node(arity, dim_out, dim_out, activation_function,
                 optimizer=Optimizer())
    node1 = Node(arity, dim_in, dim_out, activation_function,
                 optimizer=Optimizer())
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
    loss_function = MSEMultivariate()

    optimization_args = OptimizationArgs(
        inputs=input_data,
        desired_output=desired_output,
        loss_function=loss_function,
        learning_rate=0.1,
        max_iter=10000,
        min_error=sys.maxsize
    )
    output_before = [node1(input_data_i) for input_data_i in input_data]
    print("Weights before optimization:")
    print([[w, w.w_index] for w in node1.get_weights()])
    print(loss_function.evaluate(node1, input_data, desired_output))
    node1.optimize(optimization_args)
    print("Weights after optimization:")
    print([[w, w.w_index] for w in node1.get_weights()])
    print(loss_function.evaluate(node1, input_data, desired_output))
    output = [node1(input_data_i) for input_data_i in input_data]

    print("            Input         :", input_data)
    print("Before Optimization output:", output_before)
    print("       Final output       :", output)
    print("       Desired output     :", desired_output)


if __name__ == "__main__":
    main()
