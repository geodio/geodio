import sys

import numpy as np

from geodio.core.cell.operands.operand import Operand
from geodio.core.cell.operands.utility import get_predicted
from geodio.core.cell.train.loss.loss_function import LossFunction
from geodio.core.utils import flatten


class MSE(LossFunction):
    def compute_fitness(self, desired_output, predicted):
        # Mean Squared Error (MSE) fitness function
        x = np.mean(self.get_y_minus_predicted(desired_output, predicted) ** 2)
        if str(x) == "nan" or str(x) == 'inf':
            x = sys.maxsize / 2
        return x

    def compute_d_fitness(self, Y, predicted):
        return predicted - Y

    def gradient(self, cell: Operand, inputs, desired_output, index, by_weight=True):
        predicted = get_predicted(inputs, cell)
        delta_f_w_j = cell.derive(index, by_weight)
        gradient_results = np.array([delta_f_w_j(X_i) for X_i in inputs])
        result = self.compute_gradient(desired_output, gradient_results, predicted)
        return result

    def compute_gradient(self, desired_output, gradient_results, predicted):
        desired_output = flatten(desired_output)
        predicted = flatten(predicted)
        gradient_results = flatten(gradient_results)[:len(desired_output)]
        per_i = - self.get_y_minus_predicted(desired_output, predicted) * gradient_results
        result = 2 * np.mean(per_i)
        if str(result) == "nan" or str(
                result) == 'inf' or result == np.inf:
            result = 0.0
        return result
