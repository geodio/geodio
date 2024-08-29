from core.cell.train.optimization_args import OptimizationArgs
from core.cell.train.optimizer.optimization.default_optimization import \
    Optimization
from core.cell.operands.operand import Operand


class Optimizer:

    def __init__(self):
        self.risk = False

    def __call__(self, cell: Operand,
                 args: OptimizationArgs):
        """
        Optimize a cell

        :param cell: The model or neural network cell to be optimized.
        :param args: The arguments used in the optimization.
        """
        optimizer = self.make_optimizer(cell, args)
        optimizer.optimize()

    def make_optimizer(self, cell, optim_args, ewc_lambda=0.0,
                       l2_lambda=0.0):
        optim_args = optim_args.clone()
        optimizer = Optimization(cell, optim_args, self.risk,
                                 ewc_lambda=ewc_lambda,
                                 l2_lambda=l2_lambda)
        return optimizer

    def clone(self):
        cloned = Optimizer()
        cloned.risk = self.risk
        return cloned


