from typing import List

from core.cell.cell import Cell
from core.organism.layer import Layer, LayerType
from core.organism.link import Link
from core.cell.optim.loss import LossFunction
from core.cell.optim.optimization_args import OptimizationArgs


class Organism(Cell):
    def __init__(self, layers: List[Layer], arity):
        super().__init__(layers[0], arity, len(layers))
        self.layers = layers
        self.cell_map = {}  # Mapping from cell ID to actual Cell instance
        self.cell_id_to_layer_id = {}
        self.__fill_maps()
        self.layers[0].layer_type = LayerType.INPUT
        self.layers[-1].layer_type = LayerType.OUTPUT

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
        print("CELL_ID", from_id, "\n\tLINKED_IDX", to_ids)
        from_cell = self.get_cell_by_id(from_id)
        linked_cells = [self.get_cell_by_id(to_id) for to_id in to_ids]
        layer: Layer = self.layers[self.cell_id_to_layer_id[from_id]]
        layer.link(from_cell, linked_cells)

    def clone(self) -> "Organism":
        cloned_layers = [layer.clone() for layer in self.layers]
        cloned_organism = Organism(cloned_layers, self.arity)
        return cloned_organism

    def to_python(self) -> str:
        ret = ""
        for i, layer in enumerate(self.layers):
            ret += f"\n\tLAYER {i}\n"
            for cell in layer.children:
                if not isinstance(cell, Link):
                    ret += f"{cell.id} => {cell.to_python()}\n"
                else:
                    ret += f"{cell.to_python()}\n"
        return ret

    def optimize_values(self, fit_fct: LossFunction, variables,
                        desired_output,
                        learning_rate=0.1,
                        max_iterations=100,
                        min_error=10):

        optim_args = OptimizationArgs(
            learning_rate=learning_rate,
            max_iter=max_iterations,
            min_fitness=min_error,
            fitness_function=fit_fct
        )

        for x, y in zip(variables, desired_output):
            self(x)
            print("ORGANISM_INPUT", x)
            optim_args.desired_output = y
            optim_args.inputs = x
            self.layered_optimization(optim_args)
            outputs = [self(inputs) for inputs in variables]
            self.error = fit_fct(outputs, desired_output)
            print(self)
            print(self.error)
            print("DYNAMIC For input:", x,
                  "output:", self(x),
                  "desired output:", y)

    def layered_optimization(self, opt: OptimizationArgs):
        for layer in reversed(self.layers):
            layer.optimize_values(
                opt.fitness_function, opt.inputs, opt.desired_output,
                opt.learning_rate, opt.max_iter, opt.min_fitness
            )
