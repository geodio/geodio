from abc import ABC

from core.cell.train.optimization_args import OptimizationArgs
from core.cell.operands.operand import Operand
from core.cell.math.backpropagation import Backpropagatable
from core.cell.train.optimizable import OptimizableOperand


class BOO(OptimizableOperand, Backpropagatable, ABC):
    """
    Backpropagatable Optimizable Operand.
    """
    def __init__(self, children, optimizer=None):
        super().__init__(1, optimizer)
        if len(children) != 1:
            raise ValueError(f"Children of {self.__name__} must have exactly "
                             f"one element")
        self.children = children

    def optimize(self, args: OptimizationArgs):
        if args.backpropagation:
            if not isinstance(self.children[0], Backpropagatable):
                raise ValueError(f"Children of a backpropagatable "
                                 f"optimizable operand must be "
                                 f"of type Backpropagatable when "
                                 f"optimizing via backpropagation")
        self.optimizer(self, args)

    child = property(lambda self: self.children[0])

    def set_child(self, child: Operand):
        self.children[0] = child

    def get_gradients(self):
        gradients = self.get_local_gradients()
        child = self.children[0]
        gradients.extend(child.get_gradients())
        return gradients

    def __call__(self, x, meta_args=None):
        in_data = self.children[0](x, meta_args)
        return self.forward(in_data, meta_args)
