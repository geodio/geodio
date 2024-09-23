from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from core.cell.operands.weight import AbsWeight
from core.cell.operands.constant import ZERO, Operand
from core.cell.operands.variable import Variable, MetaVariable
from core.cell.train.optimizable import OptimizableOperand
from core.cell.train.optimizer import Optimizer


class Forest(OptimizableOperand):
    def __init__(self, trees: List[OptimizableOperand], arity,
                 optimizer: Optional[Optimizer] = None):
        super().__init__(arity, optimizer)
        self.children = trees

    def derive_uncached(self, index, by_weights):
        non_independent_trees = list(filter(lambda x: x.is_independent_of(
            index, by_weights), self.children))
        if len(non_independent_trees) == 0:
            return ZERO
        elif len(non_independent_trees) == 1:
            return non_independent_trees[0].derive(index, by_weights)
        else:
            derived_trees = [tree.derive(index, by_weights) for tree in
                             self.children]
            return Forest(derived_trees, self.arity, self.optimizer.clone())

    def __call__(self, args, meta_args=None):
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(tree, args, meta_args): idx for
                       idx, tree in enumerate(self.children)}
            results = [None] * len(self.children)

            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

            return results

    def clone(self) -> "Operand":
        cloned_trees = [tree.clone() for tree in self.children]
        return Forest(cloned_trees, self.arity, self.optimizer.clone())

    def to_python(self) -> str:
        pass


def forest_derive(operand_to_derive: Operand, operands: List[Operand]):
    derivatives = []
    for operand in operands:
        if isinstance(operand, Variable):
            derivative = operand_to_derive.derive(operand.value, False)
        elif isinstance(operand, AbsWeight):
            derivative = operand_to_derive.derive(operand.w_index, True)
        elif isinstance(operand, MetaVariable):
            derivative = operand_to_derive.derive(operand.meta_id, False)
        else:
            continue
        derivatives.append(derivative)
    return Forest(derivatives, operand_to_derive.arity)
