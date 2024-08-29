import sys
from typing import List

import numpy as np

from core.cell.train.optimizable import multi_tree_derive
from core.cell.operands.operand import Operand
from core.cell.operands.utility import get_predicted
from core.cell.train.loss.loss_function import LossFunction
from core.utils import flatten


class MSE(LossFunction):
    def compute_fitness(self, Y, predicted):
        # Mean Squared Error (MSE) fitness function
        x = np.mean(self.get_y_minus_predicted(Y, predicted) ** 2)
        if str(x) == "nan" or str(x) == 'inf':
            x = sys.maxsize / 2
        return x

    def compute_d_fitness(self, Y, predicted):
        return predicted - Y

    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        predicted = get_predicted(X, cell)
        delta_f_w_j = cell.derive(index, by_weight)
        gradient_results = np.array([delta_f_w_j(X_i) for X_i in X])
        result = self.compute_gradient(Y, gradient_results, predicted)
        return result

    def multi_gradient(self, cell, X, Y,
                       operands: List[Operand]):
        m_tree = multi_tree_derive(cell, operands)
        Y_flat = [y[0] for y in Y]

        predicted = get_predicted(X, cell)
        m_gradient_results = np.array([m_tree(x_i) for x_i in X]).T
        return [
            self.compute_gradient(Y_flat, gradient_results, predicted)
            for gradient_results in m_gradient_results
        ]

    def compute_gradient(self, Y, gradient_results, predicted):
        Y = flatten(Y)
        predicted = flatten(predicted)
        gradient_results = flatten(gradient_results)[:len(Y)]
        per_i = - self.get_y_minus_predicted(Y, predicted) * gradient_results
        result = 2 * np.mean(per_i)
        if str(result) == "nan" or str(
                result) == 'inf' or result == np.inf:
            result = 0.0
        return result
