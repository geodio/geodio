from typing import List

from core.cell.cell import Cell
from core.cell.optim.optimization_args import OptimizationArgs
from core.organism.layer import Layer, LayerType
from core.organism.link import Link


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
        for layer in self.layers:
            layer(args)
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
        new_link = layer.link(from_cell, linked_cells)
        if new_link is not None:
            self.cell_map[from_id] = new_link

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

    def optimize(self, optim_args: OptimizationArgs):

        optim_args_clone = optim_args.clone()
        past_x = None
        past_y = None
        for x, y in zip(optim_args.inputs, optim_args.desired_output):
            self(x)
            print("ORGANISM_INPUT", x)
            if past_x is None:
                optim_args_clone.desired_output = [y]
                optim_args_clone.inputs = [x]
            else:
                optim_args_clone.desired_output = [y, past_y]
                optim_args_clone.inputs = [x, past_x]
            second_clone = optim_args_clone.clone()
            self.layered_optimization_backwards(optim_args_clone)
            self.layered_optimization_forward(second_clone)
            self.mark_checkpoint()
            optim_args.actual_output = [
                self(inputs) for inputs in optim_args.inputs
            ]
            self.error = optim_args.compute_error()
            print(self)
            print(', '.join([
                f"<{a}, {b}" for a, b in
                zip(optim_args.desired_output, optim_args.actual_output)
            ]))
            print(self.error)
            print("DYNAMIC For input:", x,
                  "output:", self(x),
                  "desired output:", y)
            past_x = x
            past_y = y

    def layered_optimization_backwards(self, opt: OptimizationArgs):
        i = len(self.layers)
        for layer in reversed(self.layers):
            print("BACKWARD ", i)
            i -= 1
            layer.optimize(opt)

    def layered_optimization_forward(self, opt: OptimizationArgs):
        for layer in self.layers:
            for cell in layer.children:
                cell.get_state_weight().lock()
        i = 0
        for layer in self.layers:
            print("FORWARD ", i)
            i += 1
            layer.optimize(opt)

        for layer in self.layers:
            for cell in layer.children:
                cell.get_state_weight().unlock()

    def mark_checkpoint(self):
        for layer in self.layers:
            layer.mark_checkpoint()
