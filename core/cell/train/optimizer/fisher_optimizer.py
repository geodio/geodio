import numpy as np

from core.cell.train.optimization_args import OptimizationArgs
from core.cell.operands.operand import Operand
from core.cell.train.optimizer.default_optimizer import Optimizer


class FisherOptimizer(Optimizer):

    def __init__(self):
        super().__init__()
        self.risk = True
        self.fisher_information = None

    def __call__(self, cell: Operand, opt: OptimizationArgs):
        """
        Optimize a cell

        :param cell: The model or neural network cell to be optimized.
        :param args: The arguments used in the optimization.
        """
        optimizer = self.make_optimizer(
            cell, opt, ewc_lambda=0.05, l2_lambda=0.001
        )
        if self.fisher_information is None:
            self.fisher_information = np.ones(len(cell.get_weights()))
        else:
            optimizer.update_ewc_importance(self.fisher_information)
        optimizer.optimize()
        self.fisher_information = calculate_fisher_information(
            cell, opt.inputs, self.fisher_information
        )

    def clone(self):
        cloned = FisherOptimizer()
        cloned.risk = self.risk
        cloned.fisher_information = self.fisher_information
        return cloned


def calculate_fisher_information(cell, inputs, old_fisher):
    """
    Calculate the Fisher Information Matrix.

    :param cell: The model or neural network cell.
    :param inputs: The input data.
    :param old_fisher: The Old Fisher Information Matrix.
    :return: The New Fisher Information Matrix.
    """
    fisher_information = old_fisher

    # TODO
    for i, weight in enumerate(cell.get_weights()):
        grad = np.mean(np.array(cell.derive(i)(inputs)))
        # derive
        # method
        fisher_information[i] += np.square(grad)

    return fisher_information / len(inputs)
