import sys
from abc import ABC, ABCMeta, abstractmethod
from typing import Optional

from core.cell.operands.operand import Operand
from core.cell.optim.loss import LossFunction
from core.cell.optim.optimizer import Optimizer


class Optimizable(ABC, metaclass=ABCMeta):

    @abstractmethod
    def optimize_values(self, fit_fct: LossFunction, variables,
                        desired_output,
                        learning_rate=0.1,
                        max_iterations=100,
                        min_error=sys.maxsize):
        pass


class OptimizableOperand(Operand, Optimizable, metaclass=ABCMeta):

    def __init__(self, arity, optimizer: Optional[Optimizer] = None):
        super().__init__(arity)
        if optimizer is None:
            self.optimizer = Optimizer()
        else:
            self.optimizer = optimizer

    def set_optimization_risk(self, risk: bool):
        self.optimizer.risk = risk
