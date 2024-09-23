from abc import ABCMeta
from typing import Callable

from core.cell.operands.utility import get_predicted
from core.cell.train.optimization_args import OptimizationArgs
from core.cell.train.optimization.regularization \
    import BaseGradReg, AdamGradReg
from core.cell.operands.weight import t_weight


def make_grad_reg_builder(grad_reg_type: str) -> Callable[
    [t_weight, OptimizationArgs], BaseGradReg]:
    if grad_reg_type == 'adam':
        return lambda weight, args: AdamGradReg(weight, args)
    else:
        return lambda weight, args: BaseGradReg(weight, args)


class BaseOptimization(metaclass=ABCMeta):
    def __init__(self, cell, weights, optim_args: OptimizationArgs):
        """
        Initialize the optimization process, store optimization arguments, and prepare weights.

        :param cell: The computational unit (e.g., neural network).
        :param weights: The weights to optimize.
        :param optim_args: Optimization arguments (cloned).
        """
        self.optim_args = optim_args.clone()
        self.weights = weights
        self.prev_weights = [weight.get() for weight in self.weights]
        self.cell = cell
        grad_reg_builder = make_grad_reg_builder(optim_args.grad_reg)
        self.regularizers = [
            grad_reg_builder(weight, self.optim_args) for weight in
            self.weights
        ]
        self.prev_error = cell.error

    def optimize(self):
        """
        Perform optimization over the specified number of iterations.
        """
        for iteration in range(self.optim_args.max_iter):
            gradients = self.calculate_gradients()
            self.update_weights(gradients)
            self.cell.error = self.optim_args.loss_function(
                self.optim_args.desired_output,
                get_predicted(self.optim_args.inputs, self.cell)
            )
            if self.cell.error < 1e-20:
                break

    def calculate_gradients(self):
        """
        Calculate the gradients for each weight.

        :return: A list of gradients for each weight.
        """
        return [
            self.calculate_gradient(j) for j in range(len(self.weights))
        ]

    def calculate_gradient(self, j):
        """
        Calculate the gradient for the j-th weight.

        :param j: Index of the weight.
        :return: Calculated gradient for the weight.
        """
        return self.optim_args.loss_function.gradient(
            self.cell, self.optim_args.inputs, self.optim_args.desired_output,
            j
        )

    def update_weights(self, gradients):
        """
        Update the weights based on the calculated gradients using regularization.

        :param gradients: List of gradients for each weight.
        """
        for i, weight in enumerate(self.weights):
            if weight.is_locked:
                continue
            gradient = gradients[i]
            new_error = self.regularizers[i].apply_regularization(
                gradient, self.cell)
            self.cell.error = new_error

    def update_optim_args(self, optim_args: OptimizationArgs):
        self.optim_args = optim_args.clone()
        for regularizer in self.regularizers:
            regularizer.optim_args = self.optim_args
