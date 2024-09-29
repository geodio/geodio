import sys
from abc import ABC, ABCMeta, abstractmethod
from typing import Optional

from geodio.core.cell.operands.operand import Operand
from geodio.core.cell.train.optimization_args import OptimizationArgs
from geodio.core.cell.train.optimizer import Optimizer


class Optimizable(ABC, metaclass=ABCMeta):

    @abstractmethod
    def optimize(self, args: OptimizationArgs):
        pass

    def optimize_values(self, fit_fct, variables,
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
        self.derivative_cache = {}
        if optimizer is None:
            self.optimizer = Optimizer()
        else:
            self.optimizer = optimizer

    def set_optimization_risk(self, risk: bool):
        self.optimizer.risk = risk

    def __invert__(self):
        pass

    def get_weights_local(self):
        weights = []
        for child in self.get_sub_operands():
            weights.extend(child.get_weights_local())
        return weights

    def set_weights(self, new_weights):
        past_weights = self.get_weights()
        for new, past in zip(new_weights, past_weights):
            new.set(past)

    def get_sub_operands(self):
        return self.children

    def derive(self, index, by_weights=True):
        derivative_id = f'{"W" if by_weights else "X"}_{index}'
        if derivative_id not in self.derivative_cache:
            derivative = self.derive_uncached(index, by_weights)
            self.derivative_cache[derivative_id] = derivative
        return self.derivative_cache[derivative_id]

    @abstractmethod
    def derive_uncached(self, index, by_weights):
        pass

    def optimize(self, args: OptimizationArgs):
        self.optimizer(self, args)
