import sys
from abc import ABC, ABCMeta, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List

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
        for child in self.get_sub_items():
            weights.extend(child.get_weights_local())
        return weights

    def set_weights(self, new_weights):
        past_weights = self.get_weights()
        for new, past in zip(new_weights, past_weights):
            new.set(past)

    def get_sub_items(self):
        return self.children

    def derive(self, index, by_weights=True):
        derivative_id = f'{"W" if by_weights else "X"}_{index}'
        if derivative_id not in self.derivative_cache:
            derivative = self.derive_unchained(index, by_weights)
            self.derivative_cache[derivative_id] = derivative
        return self.derivative_cache[derivative_id]

    @abstractmethod
    def derive_unchained(self, index, by_weights):
        pass

    def optimize(self, args: OptimizationArgs):
        self.optimizer(self, args)

    def multi_tree_derive(self, by_weights=True):

        keys = list(filter(
            lambda x: x.startswith("W" if by_weights else "X"),
            self.derivative_cache.keys()
        ))


class MultiTree(OptimizableOperand):
    def __init__(self, trees: List[OptimizableOperand], arity, optimizer: Optional[Optimizer] = None):
        super().__init__(arity, optimizer)
        self.children = trees

    def derive_unchained(self, index, by_weights):
        derived_trees = [tree.derive(index, by_weights) for tree in self.children]
        return MultiTree(derived_trees, self.arity, self.optimizer.clone())

    def __call__(self, args, meta_args=None):
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    tree,
                    args
                ): tree for tree in self.children
            }
            for future in as_completed(futures):
                future.result()

    def clone(self) -> "Operand":
        cloned_trees = [tree.clone() for tree in self.children]
        return MultiTree(cloned_trees, self.arity, self.optimizer.clone())

    def to_python(self) -> str:
        pass
