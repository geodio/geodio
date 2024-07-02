from typing import Dict, List, TypeVar, Any

from src.cell.cell import Cell, t_cell
from src.cell.operands.operand import Operand
from src.cell.optim.fitness import FitnessFunction

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

    def optimize_values(self, fit_fct: FitnessFunction, variables,
                        desired_output,
                        learning_rate=0.1, max_iterations=100, min_fitness=10):
        """
        Optimize the values of the cells based on the fitness function.
        """
        for layer in self.layers.values():
            for cell in layer.values():
                if cell.fitness is None:
                    cell.optimize_values(fit_fct, variables, desired_output,
                                         learning_rate, max_iterations,
                                         min_fitness)

    def __call__(self, inputs):
        self.update_states_forward(inputs)
        self.update_states_backward(inputs)
        return [self.cells[cell_id].state for cell_id in self.layers[0]]
