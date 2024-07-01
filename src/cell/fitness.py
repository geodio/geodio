import sys
from abc import ABC, abstractmethod

import numpy as np

from src.cell.operands.operand import Operand


def get_predicted(X, cell):
    return [cell(x_inst) for x_inst in X]


class FitnessFunction(ABC):

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


class MSE(FitnessFunction):
    def compute_fitness(self, Y, predicted):
        # Mean Squared Error (MSE) fitness function
        x = np.mean((Y - predicted) ** 2)
        if str(x) == "nan" or str(x) == 'inf':
            x = sys.maxsize / 2
        return x

    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        predicted = get_predicted(X, cell)
        delta_f_w_j = cell.derive(index, by_weight)
        per_i = (Y - predicted) * [delta_f_w_j(X_i) for X_i in X]
        result = 2 * np.mean(per_i)
        if str(result) == "nan" or str(result) == 'inf' or result == np.inf:
            result = 0.0
        return result
