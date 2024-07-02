import random
from abc import ABC, abstractmethod
from typing import Optional, List, Dict

from src.genetic.organism import Organism, t_layer_id, t_cell_ids
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


class RandomOrganismGenerator(OrganismGenerator):

    def generate_organism(self, depth=None) -> Organism:
        if self.organism is None:
            self.organism = Organism(self.max_depth, self.arity)
        if depth is None:
            depth = self.max_depth
        layers, connections = self._generate_random_shape(depth)
        id_mapping: Dict[int, int] = {}

        for layer_id in range(len(layers)):
            layer = layers[layer_id]
            for cell_id in layer:
                cell_arity = len(connections[cell_id])
                if layer_id == 0:
                    cell_arity += self.arity
                self.cell_generator.set_arity(cell_arity)
                new_cell = self.cell_generator.new_offspring()
                map_cell_id = self.organism.add_cell(layer_id, new_cell)
                id_mapping[cell_id] = map_cell_id
        for cell_id, links in connections.items():
            for to_id in links:
                self.organism.connect_cells(id_mapping[cell_id],
                                            id_mapping[to_id])
        result = self.organism
        self.organism = None
        return result

    def _generate_random_shape(self, depth):
        layers: Dict[t_layer_id, t_cell_ids] = {i: [] for i in range(depth + 1)}
        k = 0
        for layer_id in range(depth + 1):
            if layer_id == 0:
                per_layer = self.output_arity
            else:
                per_layer = random.randint(1, self.mcpl)
            for i in range(per_layer):
                layers[layer_id].append(k)
                k += 1

        connections = {cell_id: [] for layer in layers.values() for cell_id in layer}

        for layer_id in range(depth + 1):
            for cell_id in layers[layer_id]:
                if layer_id > 0:
                    num_lower_layer_connections = random.randint(1, self.llcr)
                    for _ in range(num_lower_layer_connections):
                        to_layer = layer_id - 1
                        add_connection(cell_id, connections,
                                       layers, to_layer)

                if layer_id < depth:
                    num_higher_layer_connections = random.randint(1, self.hlcr)
                    for _ in range(num_higher_layer_connections):
                        to_layer = layer_id + 1
                        add_connection(cell_id, connections,
                                       layers, to_layer)

                num_equal_layer_connections = random.randint(1, self.elcr)
                for _ in range(num_equal_layer_connections):
                    add_connection(cell_id, connections,
                                   layers, layer_id)

        return layers, connections

