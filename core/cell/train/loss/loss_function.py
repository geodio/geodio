from abc import ABC, ABCMeta, abstractmethod

import numpy as np

from core.cell.operands.utility import get_predicted
from core.cell.operands.operand import Operand

from core.utils import flatten


class LossFunction(ABC, metaclass=ABCMeta):

    def evaluate(self, cell: Operand, X, Y):
        predicted = get_predicted(X, cell)
        return self.compute_fitness(flatten(Y), flatten(predicted))

    @abstractmethod
    def compute_fitness(self, Y, predicted):
        pass

    @abstractmethod
    def compute_d_fitness(self, Y, predicted):
        pass

    def __call__(self, Y, predicted):
        return self.compute_fitness(flatten(Y), flatten(predicted))

    @abstractmethod
    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        pass

    def get_y_minus_predicted(self, Y, predicted):
        Y_minus_predicted = np.array(Y) - np.array(predicted[:len(Y)])
        return Y_minus_predicted
