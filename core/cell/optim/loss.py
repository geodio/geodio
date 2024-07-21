import sys
from abc import ABC, abstractmethod

import numpy as np

from core.cell.operands.operand import Operand
from core.cell.operands.stateful import Stateful
from core.utils import flatten


def get_predicted(X, cell):
    return flatten([cell(x_inst) for x_inst in X])


class LossFunction(ABC):

    def evaluate(self, cell: Operand, X, Y):
        predicted = get_predicted(X, cell)
        return self.compute_fitness(flatten(Y), flatten(predicted))

    @abstractmethod
    def compute_fitness(self, Y, predicted):
        pass

    def __call__(self, Y, predicted):
        return self.compute_fitness(flatten(Y), flatten(predicted))

    @abstractmethod
    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        pass

    def get_y_minus_predicted(self, Y, predicted):
        Y_minus_predicted = np.array(Y) - np.array(predicted[:len(Y)])
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
        result = self.compute_gradient(Y, gradient_results, predicted)
        return result

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


class CheckpointedMSE(MSE):

    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        assert isinstance(cell, Stateful) and isinstance(cell, Operand)
        predicted = get_predicted(X[0], cell)
        delta_f_w_j = cell.derive(index, by_weight)
        gradient_results = [delta_f_w_j(X_i) for X_i in X[0]]

        if len(X) > 1:
            cell.use_checkpoint()
            predicted.extend(get_predicted(X[1], cell))
            gradient_results.extend([delta_f_w_j(X_i) for X_i in X[1]])
            gradient_results = np.array(gradient_results)
            cell.use_current()

        # print("LOSS_CHECKPOINT_MSE_GRADIENT", X, Y, predicted)
        result = self.compute_gradient(Y, gradient_results, predicted)
        return result
