from core.cell.train.optimization_args import OptimizationArgs
from core.cell.train.optimization.default_optimization import \
    Optimization
from core.cell.train.optimization import \
    BackpropagationOptimization
from core.cell.operands.operand import Operand


class Optimizer:

    def __init__(self, optimization=None):
        self.risk = False
        self.optimization = optimization

    def __call__(self, cell: Operand,
                 args: OptimizationArgs):
        """
        Optimize a cell

        :param cell: The model or neural network cell to be optimized.
        :param args: The arguments used in the optimization.
        """

        optimizer = self.make_optimizer(cell, args)
        optimizer.optimize()

    def make_optimizer(self, cell, optim_args):
        optim_args = optim_args.clone()
        if self.optimization == "backpropagation" or optim_args.backpropagation:
            optimizer = BackpropagationOptimization(cell, optim_args)
        else:
            optimizer = Optimization(cell, optim_args)
        return optimizer

    def clone(self):
        cloned = Optimizer()
        cloned.risk = self.risk
        return cloned
