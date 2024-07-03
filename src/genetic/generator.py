import random
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from src.genetic.organism import Organism
from src.math import rnd
from src.cell.cell import Cell, crossover
from src.cell.collections.functors import Functors
from src.cell.operands.function import Function
from src.cell.operands.operand import Operand
from src.cell.operands.variable import Variable
from src.cell.operands.weight import Weight
from src.genetic.pop_utils import PopulationProperties, ReproductionPolicy


class CellGenerator(ABC):
    def __init__(self):
        self.pop_prop: Optional[PopulationProperties] = None

    def set_pop_prop(self, pop_prop: PopulationProperties):
        self.pop_prop = pop_prop

    def set_arity(self, arity):
        self.pop_prop.arity = arity

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


class RandomCellGenerator(CellGenerator):
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


def add_connection(cell_id, connections, layers, to_layer):
    if layers[to_layer]:
        to_id = random.choice(layers[to_layer])
        connections[cell_id].append(to_id)


class Link(Functors):
    def __init__(self, from_cell: Cell, to_cell: Cell):
        self.from_cell = from_cell
        self.to_cell = to_cell

    def update_state(self):
        self.to_cell.state = self.from_cell.state


class RandomOrganismGenerator(OrganismGenerator):
    def generate_organism(self, depth=None) -> Organism:
        if depth is None:
            depth = self.max_depth

        cells_per_layer = min(self.mcpl, self.arity)
        layers = [[] for _ in range(depth)]
        connections: Dict[int, List[int]] = {}

        # Generate cells layer by layer
        cell_count = 0
        for d in range(depth):
            for _ in range(cells_per_layer):
                new_cell = self.cell_generator.new_offspring()
                new_cell.id = cell_count
                layers[d].append(cell_count)
                connections[cell_count] = []
                cell_count += 1

        # Create an Organism
        organism = Organism(layers[0][0], self.max_depth, self.arity)
        organism.layers = layers

        # Link cells within the organism
        for layer_idx, layer in enumerate(layers):
            for cell_id in layer:
                # Higher layer connections
                if layer_idx > 0:
                    for _ in range(random.randint(*self.hlcr)):
                        add_connection(cell_id, connections, layers, layer_idx - 1)

                # Equal layer connections
                for _ in range(random.randint(*self.elcr)):
                    add_connection(cell_id, connections, layers, layer_idx)

                # Lower layer connections
                if layer_idx < depth - 1:
                    for _ in range(random.randint(*self.llcr)):
                        add_connection(cell_id, connections, layers, layer_idx + 1)

        # Update cell connections in the organism
        for from_id, to_ids in connections.items():
            from_cell = organism.get_cell_by_id(from_id)
            for to_id in to_ids:
                to_cell = organism.get_cell_by_id(to_id)
                link = Link(from_cell, to_cell)
                link.update_state()

        return organism

