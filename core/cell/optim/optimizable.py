import sys
from abc import ABC, ABCMeta, abstractmethod
from typing import Optional

from core.cell.operands.operand import Operand
from core.cell.optim.loss import LossFunction
from core.cell.optim.optimization_args import OptimizationArgs
from core.cell.optim.optimizer import Optimizer


class Optimizable(ABC, metaclass=ABCMeta):

    @abstractmethod
    def optimize(self, args: OptimizationArgs):
        pass

    def optimize_values(self, fit_fct: LossFunction, variables,
                        desired_output,
                        learning_rate=0.1,
                        max_iterations=100,
                        min_error=sys.maxsize):
        args = OptimizationArgs(
            loss_function=fit_fct,
            inputs=variables,
            desired_output=desired_output,
            learning_rate=learning_rate,
            max_iter=max_iterations,
            min_error=min_error
        )
        self.optimize(args)


class OptimizableOperand(Operand, Optimizable, metaclass=ABCMeta):

    def __init__(self, arity, optimizer: Optional[Optimizer] = None):
        super().__init__(arity)
        if optimizer is None:
            self.optimizer = Optimizer()
        else:
            self.optimizer = optimizer

    def set_optimization_risk(self, risk: bool):
        self.optimizer.risk = risk

    def __invert__(self):
        pass

    def get_weights(self):
        weights = []
        for child in self.get_sub_items():
            weights.extend(child.get_weights())

        for i, weight in enumerate(weights):
            weight.w_index = i

        return weights

    def set_weights(self, new_weights):
        past_weights = self.get_weights()
        for new, past in zip(new_weights, past_weights):
            new.set(past)

    def get_sub_items(self):
        return self.children