from abc import ABC, abstractmethod

import numpy as np

from core.cell.optim.optimizable import OptimizableOperand
from core.genetic.pop_utils import ReproductionPolicy


class GeneExpressedOptimizableOperand(OptimizableOperand, ABC):
    def __init__(self, arity: int, max_depth,
                 reproduction_policy=ReproductionPolicy.DIVISION,
                 optimizer=None):
        super().__init__(arity, optimizer)
        self.reproduction_policy = reproduction_policy
        self.depth = max_depth
        self.age = 0
        self.marked = False
        self.reproduction_policy = reproduction_policy
        self.mutation_risk = 0.5
        self.frozen = None

    @abstractmethod
    def nodes(self):
        """
        Get the nodes of this cell.
        :return: List of Operands corresponding to the first level of nodes
        from the root.
        :rtype: list
        """
        pass

    @abstractmethod
    def replace(self, node_old, node_new):
        pass

    def mutate(self, generator, max_depth=None):
        if not max_depth:
            max_depth = self.depth
        r = np.random.rand()
        if r < self.mutation_risk and len(self.nodes()) != 0:
            mutant_node = generator.generate_node(max_depth - 1)
            self.depth = max_depth
            self.randomly_replace(mutant_node)
            self.clean()
        return self

    @abstractmethod
    def clean(self):
        pass

    @abstractmethod
    def randomly_replace(self, mutant_node):
        pass

    def get_age(self):
        return self.age

    def inc_age(self, age_benefit=0):
        """
        ages the tree if the fitness exists
        :param age_benefit: contribution of age to the fitness
        :return:
        """
        if self.error is not None:
            self.age += 1
            self.error *= (1 - age_benefit)
            self.mutation_risk *= (1 - age_benefit)

    def mark(self):
        self.marked = True




