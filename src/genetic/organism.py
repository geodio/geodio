from typing import List

from src.cell.cell import Cell
from src.cell.layer import Layer
from src.cell.optim.fitness import FitnessFunction


class Organism(Cell):
    def __init__(self, layers: List[Layer], arity):
        super().__init__(layers[0], arity, len(layers))
        self.layers = layers
        self.cell_map = {}  # Mapping from cell ID to actual Cell instance
        self.cell_id_to_layer_id = {}
        self.__fill_maps()

    def __fill_maps(self):
        cell_id = 0
        for layer_id, layer in enumerate(self.layers):
            for cell in layer.children:
                self.cell_map[cell_id] = cell
                self.cell_id_to_layer_id[cell_id] = layer_id
                cell.id = cell_id
                cell_id += 1

    def get_cell_by_id(self, cell_id: int) -> Cell:
        return self.cell_map.get(cell_id)

    def __call__(self, args):
        # Evaluate the organism layer by layer
        inputs = args
        for layer in self.layers:
            layer(inputs)
        return self.layers[-1].get_cell_states()

    def add_cell(self, cell: Cell, layer_idx: int):
        if layer_idx >= len(self.layers):
            raise IndexError("Layer index out of range")
        cell_id = len(self.cell_map)
        self.cell_map[cell_id] = cell
        self.layers[layer_idx].add_cell(cell)
        self.cell_id_to_layer_id[cell_id] = layer_idx

    def connect_cells(self, from_id: int, to_ids: List[int]):
        from_cell = self.get_cell_by_id(from_id)
        linked_cells = [self.get_cell_by_id(to_id) for to_id in to_ids]
        layer: Layer = self.layers[self.cell_id_to_layer_id[from_id]]
        layer.link(from_cell, linked_cells)

    def clone(self) -> "Organism":
        cloned_layers = [layer.clone() for layer in self.layers]
        cloned_organism = Organism(cloned_layers, self.arity)
        return cloned_organism

    def to_python(self) -> str:
        return (f"organism(["
                f"{', '.join(str(layer) for layer in self.layers)}])")

    def optimize_values(self, fit_fct: FitnessFunction, variables,
                        desired_output,
                        learning_rate=0.1,
                        max_iterations=100,
                        min_fitness=10):
        desired_output = [desired_output]
        d_o = desired_output
        vars = variables
        for layer in reversed(self.layers):
            layer.optimize_values(
                fit_fct, vars, desired_output,
                learning_rate, max_iterations, min_fitness
            )
            desired_output = None
            vars = None
        outputs = [self(inputs) for inputs in variables]
        self.fitness = fit_fct(outputs, d_o)

