from abc import ABC, abstractmethod
from typing import List

import numpy as np

from core.cell.operands.operand import Operand
from core.cell.optim.optimizable import OptimizableOperand
from core.cell.optim.optimization_args import OptimizationArgs
from core.cell.optim.optimizer import Optimizer, Optimization


class Backpropagatable(OptimizableOperand, ABC):
    def __init__(self, arity, optimizer=None):
        if optimizer is None:
            optimizer = BPOptimizer()
        super().__init__(arity, optimizer)

    @abstractmethod
    def backpropagation(self, dx: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def get_gradients(self) -> List[np.ndarray]:
        pass

    def optimize(self, args: OptimizationArgs):
        self.optimizer(self, args)


class BPOptimization(Optimization):
    def __init__(self, cell: Backpropagatable, optim_args: OptimizationArgs,
                 decay_rate: float = 0.99999,
                 risk: bool = False,
                 ewc_lambda: float = 0.1,
                 l2_lambda: float = 0.0):
        super().__init__(cell, optim_args, decay_rate=decay_rate, risk=risk,
                         ewc_lambda=ewc_lambda, l2_lambda=l2_lambda)
        self.forward_input = [np.array([x[0] for x in optim_args.inputs]).T]
        self.y = np.array([y[0] for y in optim_args.desired_output]).T

    def calculate_gradients(self):
        predicted = self.cell(self.forward_input)
        # print("GRADIENTS", self.y.shape, predicted.shape)
        dz = self.fit_func.compute_d_fitness(self.y, predicted)
        # print("DZ", dz.shape)
        self.cell.backpropagation([dz])
        gradients = self.cell.get_gradients()
        print(gradients)
        return gradients


class BPOptimizer(Optimizer):
    def __init__(self):
        super().__init__()

    def make_optimizer(self, cell, optim_args, ewc_lambda=0.0,
                       l2_lambda=0.0):
        optim_args = optim_args.clone()
        optimizer = BPOptimization(cell, optim_args, self.risk,
                                   ewc_lambda=ewc_lambda,
                                   l2_lambda=l2_lambda)
        return optimizer
