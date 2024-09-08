from core.cell.train.optimization_args import OptimizationArgs
from core.cell.train.optimizer.optimization.base_optimization import \
    BaseOptimization


class Optimization(BaseOptimization):
    def __init__(self, cell, optim_args: OptimizationArgs):
        """
        Initialize the Optimization object.

        :param cell: The model or neural network cell to be optimized.
        :param optim_args: the optimization arguments.
        """
        super().__init__(cell, cell.get_weights(), optim_args)
