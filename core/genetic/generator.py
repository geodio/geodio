import random
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple

from core.cell.collections.bank import CellBank
from core.organism.organism import Organism
from core.math import rnd
from core.cell.cell import Cell
from core.genetic.cell_utils import crossover
from core.cell.collections.basefunctions import BaseFunctions
from core.cell.operands.function import Function
from core.cell.operands.operand import Operand
from core.cell.operands.variable import Variable
from core.cell.operands.weight import Weight
from core.genetic.pop_utils import PopulationProperties, ReproductionPolicy


class CellGenerator(ABC):
    def __init__(self):
        self.pop_prop: Optional[PopulationProperties] = None

    def set_pop_prop(self, pop_prop: PopulationProperties):
        self.pop_prop = pop_prop

    def set_arity(self, arity):
        self.pop_prop.arity = arity

    @abstractmethod
    def new_offspring(self):
        pass

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


class NodeBasedCellGenerator(CellGenerator, ABC):

    @abstractmethod
    def generate_node(self, depth=None) -> Operand:
        pass

    def new_offspring(self):
        root = self.generate_node()
        return Cell(root, self.pop_prop.arity, self.pop_prop.max_depth)


class RandomCellGenerator(NodeBasedCellGenerator):
    def generate_node(self, depth=None) -> Operand:
        if depth is None:
            depth = self.pop_prop.max_depth
        if depth == 0 or random.random() < 0.33:  # Terminal node
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
        if not isinstance(self.pop_prop.func_set, BaseFunctions):
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


class CellBankGenerator(CellGenerator):
    def __init__(self, cell_bank: CellBank):
        super().__init__()
        self.cell_bank = cell_bank

    def new_offspring(self, depth=None) -> Operand:
        return self.cell_bank.get_random_clone()


class OrganismGenerator(ABC):
    def __init__(self, max_depth, arity,
                 cell_generator: CellGenerator,
                 higher_layer_connectivity_range,
                 equal_layer_connectivity_range,
                 lower_layer_connectivity_range,
                 max_cells_per_layer,
                 output_arity):
        self.max_depth = max_depth
        self.arity = arity
        self.cell_generator = cell_generator
        self.hlcr = higher_layer_connectivity_range
        self.elcr = equal_layer_connectivity_range
        self.llcr = lower_layer_connectivity_range
        self.mcpl = max_cells_per_layer
        self.output_arity = output_arity

        self.organism = None

    @abstractmethod
    def generate_organism(self, depth=None) -> Organism:
        pass


class RandomOrganismGenerator(OrganismGenerator):
    def generate_organism(self, depth=None) -> Organism:
        pass
