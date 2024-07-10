import sys
from abc import ABC, abstractmethod

import numpy as np

from core.cell.operands.operand import Operand


def get_predicted(X, cell):
    return flatten([cell(x_inst) for x_inst in X])


def flatten(lst):
    """
    Flattens a list of any dimension to a 1-dimensional list.

    Parameters:
    - lst: The list to flatten.

    Returns:
    - A flattened 1-dimensional list.
    """
    flattened_list = []

    def _flatten(sublist):
        for item in sublist:
            if isinstance(item, (list, np.ndarray)):
                _flatten(item)
            else:
                flattened_list.append(item)

    _flatten(lst)
    return flattened_list

class LossFunction(ABC):

    def evaluate(self, cell: Operand, X, Y):
        predicted = get_predicted(X, cell)
        return self.compute_fitness(Y, predicted)

    @abstractmethod
    def compute_fitness(self, Y, predicted):
        pass

    def __call__(self, Y, predicted):
        return self.compute_fitness(Y, predicted)

    @abstractmethod
    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        pass

    def get_y_minus_predicted(self, Y, predicted):
        Y_minus_predicted = np.array(Y) - np.array(predicted)
        return Y_minus_predicted


class MSE(LossFunction):
    def compute_fitness(self, Y, predicted):
        # Mean Squared Error (MSE) fitness function
        x = np.mean(self.get_y_minus_predicted(Y, predicted) ** 2)
        if str(x) == "nan" or str(x) == 'inf':
            x = sys.maxsize / 2
        return x

    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        predicted = get_predicted(X, cell)
        delta_f_w_j = cell.derive(index, by_weight)
        gradient_results = np.array([delta_f_w_j(X_i) for X_i in X])
        Y = flatten(Y)
        predicted = flatten(predicted)
        gradient_results = flatten(gradient_results)
        per_i = - self.get_y_minus_predicted(Y, predicted) * gradient_results
        result = 2 * np.mean(per_i)
        if str(result) == "nan" or str(result) == 'inf' or result == np.inf:
            result = 0.0
        # print(
        #     "id:", cell.id,
        #     "result:", result,
        #     "deriv_index:", index,
        #     "X:", X,
        #     "Y:", Y,
        #     "predicted:",  predicted,
        #     "gradient:", gradient_results
        # )
        return result
