import sys
from abc import ABC, ABCMeta, abstractmethod

from core.cell.optim.fitness import FitnessFunction


class Optimizable(ABC, metaclass=ABCMeta):
    @abstractmethod
    def optimize_values(self, fit_fct: FitnessFunction, variables,
                        desired_output,
                        learning_rate=0.1,
                        max_iterations=100,
                        min_fitness=sys.maxsize):
        pass
