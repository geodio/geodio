import random
from abc import ABC, abstractmethod
from typing import Optional, List

from src.math import rnd
from src.cell.cell import Cell, crossover
from src.cell.collections.functors import Functors
from src.cell.operands.function import Function
from src.cell.operands.operand import Operand
from src.cell.operands.variable import Variable
from src.cell.operands.weight import Weight
from src.genetic.pop_utils import PopulationProperties, ReproductionPolicy


class Generator(ABC):
    def __init__(self):
        self.pop_prop: Optional[PopulationProperties] = None

    def set_pop_prop(self, pop_prop: PopulationProperties):
        self.pop_prop = pop_prop

    @abstractmethod
    def generate_node(self, depth=None) -> Operand:
        pass

    def new_offspring(self):
        root = self.generate_node()
        return Cell(root, self.pop_prop.arity, self.pop_prop.max_depth)

    def new_offspring_list(self, size):
        return [self.new_offspring() for _ in range(size)]

    def reproduce_cells(self, population_section: List[Cell], target_size):
        to_crossover = None
        reproduction_result = []
        for cell in population_section:
            if cell.reproduction_policy == ReproductionPolicy.DIVISION:
                new_cell = cell.clone()
                new_cell.age = 0
                new_cell.mutate(self, self.pop_prop.max_depth)
                reproduction_result.append(new_cell)
            elif cell.reproduction_policy == ReproductionPolicy.CROSSOVER:
                if to_crossover is None:
                    to_crossover = cell.clone()
                    to_crossover.age = 0
                else:
                    crossed_cells = crossover(to_crossover, cell.clone())
                    reproduction_result.append(crossed_cells[0])
                    reproduction_result.append(crossed_cells[1])
                    to_crossover = None
            if len(reproduction_result) == target_size:
                return reproduction_result
        while len(reproduction_result) < target_size:
            reproduction_result.append(self.new_offspring())


class RandomGenerator(Generator):
    def generate_node(self, depth=None) -> Operand:
        if depth is None:
            depth = self.pop_prop.max_depth
        if depth == 0 or random.random() < 0.3:  # Terminal node
            operand_type = "weight" if random.random() < 0.5 else "variable"
            if operand_type == "weight":
                value = random.choice(self.pop_prop.term_set)
                return Weight(value)
            value = random.randint(0, self.pop_prop.arity - 1)
            return Variable(value)
        else:  # Function node
            node = self.generate_function_node(depth)
            return node

    def generate_function_node(self, depth):
        if not isinstance(self.pop_prop.func_set, Functors):
            func = rnd.choice(self.pop_prop.func_set)
            # Number of arguments of the function
            arity = len(func.__code__.co_varnames)
            node = Function(arity=arity, value=func)
        else:
            node = self.pop_prop.func_set.get_random_clone()
            arity = node.arity
        for _ in range(arity):
            child = self.generate_node(depth - 1)
            node.add_child(child)
        return node

    def derive(self, index, by_weight=True):
        return None
