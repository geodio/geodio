from core.cell.train.optimization_args import OptimizationArgs
from core.cell.train.optimizer.optimization.base_optimization import BaseOptimization


class Optimization(BaseOptimization):
    def __init__(self, cell, optim_args: OptimizationArgs,
                 decay_rate=0.99999, risk=False, ewc_lambda=0.1,
                 l2_lambda=0.0):
        """
        Initialize the Optimization object.

        :param cell: The model or neural network cell to be optimized.
        :param decay_rate: The rate at which the learning rate decays per
        iteration.
        :param ewc_lambda: The regularization strength for EWC.
        :param l2_lambda: The regularization strength for L2 regularization.
        """
        super().__init__(cell, cell.get_weights(), optim_args, decay_rate,
                         risk, ewc_lambda, l2_lambda)
