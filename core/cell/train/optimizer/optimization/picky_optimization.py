from core.cell.train.optimization_args import OptimizationArgs
from core.cell.train.optimizer.optimization.base_optimization import \
    BaseOptimization


class PickyOptimization(BaseOptimization):
    def __init__(self, cell, optim_args: OptimizationArgs, weights,
                 decay_rate=0.99999, risk=False, ewc_lambda=0.1,
                 l2_lambda=0.0):
        super().__init__(cell, weights, optim_args, decay_rate,
                         risk, ewc_lambda, l2_lambda)
