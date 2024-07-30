import random
from abc import abstractmethod, ABC
from typing import Dict, List, TypeVar, Any

from core.cell.cell import Cell, t_cell
from core.cell.optim.loss import LossFunction
from core.genetic.generator import CellGenerator

t_layer_id = TypeVar('t_layer_id', bound=int)
t_cell_id = TypeVar('t_cell_id', bound=int)
t_cell_ids = TypeVar('t_cell_ids', bound=List[int])
t_cells = TypeVar('t_cells', bound=List[Cell])
t_states = TypeVar('t_states', bound=List[Any])


class Organism(Cell):
    def __init__(self, max_depth, arity: int):
        super().__init__(self, arity, max_depth)
        self.layers: Dict[t_layer_id, t_cell_ids] = {
            i: [] for i in range(max_depth + 1)
        }
        self.cells: t_cells = []
        self.links: Dict[t_cell_id, t_cell_ids] = {}

    def add_cell(self, layer_id: t_layer_id,
                 cell: t_cell) -> t_cell_id:
        cell_id = len(self.cells)
        self.cells.append(cell)
        self.layers[layer_id].append(cell_id)
        self.links[cell_id] = []
        return cell_id

    def connect_cells(self, from_id: t_cell_id, to_id: t_cell_id):
        if to_id >= len(self.cells):
            raise ValueError(
                f"Cell {to_id} does not exist.")
        if from_id >= len(self.cells):
            raise ValueError(
                f"Cell {from_id} does not exist.")

        self.links[from_id].append(to_id)

    def get_cell(self, cell_id: int) -> t_cell:
        return self.cells[cell_id]

    def get_connected_cells_in_layer(self,
                                     layer_id: t_layer_id,
                                     cell_id: t_cell_id
                                     ) -> t_cell_ids:
        layer = self.layers[layer_id]
        links = self.links[cell_id]
        return [neigh_id for neigh_id in links if neigh_id in layer]

    def get_state_of_links_for_cell(self, cell_id: t_cell_id) -> t_states:
        links = self.links[cell_id]
        return [self.cells[cell_id].state for cell_id in links]

    def get_inputs_for_cell(self, layer: int, cell_id: int, inputs) -> List:
        """
        Get input values for a cell based on its layer and connections.
        """
        if layer == 0:
            return inputs + self.get_state_of_links_for_cell(cell_id)
        else:
            return self.get_state_of_links_for_cell(cell_id)

    def update_states_forward(self, inputs):
        """
        Forward phase.

        Update the states of the cells based on the input data.
        """
        for layer_id in range(len(self.layers)):
            self.update_states(inputs, layer_id)

    def update_states(self, inputs, layer_id):
        new_states = {
            cell_id: self.cells[cell_id](
                self.get_inputs_for_cell(layer_id, cell_id, inputs)
            )
            for cell_id in self.layers[layer_id]
        }

        for cell_id, new_state in new_states.items():
            cell = self.cells[cell_id]
            cell.state = new_state

    def update_states_backward(self, inputs):
        """
        Backward phase.

        Update the states of the cells based on the backward data flow.
        """
        last_layer_id = len(self.layers) - 1
        for layer_id in range(last_layer_id, -1, -1):
            if layer_id == last_layer_id:
                continue
            self.update_states(inputs, layer_id)

    def optimize(self, fit_fct: LossFunction, variables,
                 desired_output, learning_rate=0.1, max_iterations=100,
                 min_error=10):
        for iteration in range(max_iterations):
            gradients = self.calculate_gradients(fit_fct, variables,
                                                 desired_output)
            self.update_weights(gradients, learning_rate)

            fitness = self.calculate_fitness(fit_fct, variables,
                                             desired_output)
            if fitness <= min_error:
                break

            learning_rate *= 0.99999  # Decay the learning rate

    def calculate_gradients(self, fit_fct, variables, desired_output):
        gradients = []
        for layer_id, cell_ids in self.layers.items():
            layer_gradients = []
            for cell_id in cell_ids:
                cell = self.cells[cell_id]
                cell_inputs = self.get_inputs_for_cell(
                    layer_id, cell_id, variables
                )
                gradient = fit_fct.gradient(cell, cell_inputs, desired_output, 0)
                layer_gradients.append(gradient)
            gradients.append(layer_gradients)
        return gradients

    def update_weights(self, gradients, learning_rate):
        for layer_id, layer_gradients in enumerate(gradients):
            for cell_id, gradient in zip(self.layers[layer_id],
                                         layer_gradients):
                cell = self.cells[cell_id]
                for i, weight in enumerate(cell.weights):
                    cell.weights[i] = weight - learning_rate * gradient[i]

    def calculate_fitness(self, fit_fct, variables, desired_output):
        predictions = self(variables)
        return fit_fct(desired_output, predictions)

    def __call__(self, args, meta_args=None):
        self.update_states_forward(inputs)
        self.update_states_backward(inputs)
        return [self.cells[cell_id].state for cell_id in self.layers[0]]


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

