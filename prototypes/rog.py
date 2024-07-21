import random
from typing import Tuple, List

from core.cell.cell import Cell
from core.genetic.generator import OrganismGenerator, CellGenerator
from prototypes.layer_2 import DistributionPolicy, Layer
from prototypes.organism_2 import Organism


class RandomOrganismGenerator(OrganismGenerator):
    def __init__(self, max_depth: int, arity: int,
                 cell_generator: CellGenerator,
                 higher_layer_connectivity_range: Tuple[int, int],
                 equal_layer_connectivity_range: Tuple[int, int],
                 lower_layer_connectivity_range: Tuple[int, int],
                 max_cells_per_layer: int,
                 output_arity: int):
        super().__init__(max_depth, arity, cell_generator,
                         higher_layer_connectivity_range,
                         equal_layer_connectivity_range,
                         lower_layer_connectivity_range, max_cells_per_layer,
                         output_arity)

    def generate_organism(self, depth: int = None) -> Organism:
        if depth is None:
            depth = self.max_depth
        input_layer = Layer(
            self.arity, distribution_policy=DistributionPolicy.SPLIT
        )
        layers = [input_layer]
        for _ in range(depth):
            layers.append(Layer(self.arity))

        for d in range(depth + 1):
            cell_count = random.randint(1, self.mcpl)
            self.cell_generator.set_arity(1)
            if d == 0:
                cell_count = self.arity
                # self.cell_generator.set_arity(self.arity)
            elif d == depth:
                cell_count = self.output_arity
            for _ in range(cell_count):
                new_cell: Cell = self.cell_generator.new_offspring()
                new_cell.set_optimization_risk(True)
                layers[d].add_cell(new_cell)

        organism = Organism(layers, self.arity)
        print(layers)

        for layer_idx, layer in enumerate(layers):
            if layer_idx != 0:
                for cell in layer.children:
                    self.link_cell(cell, depth, layer_idx, layers, organism)

        return organism

    def link_cell(self, cell, depth, layer_idx, layers, organism):
        target_idx = []
        # Higher layer connections
        hlcr = random.randint(*self.hlcr)
        self.extend_target_idx(
            layer_idx - 1, layers, hlcr, target_idx
        )
        # Equal layer connections
        elcr = random.randint(*self.elcr)
        self.extend_target_idx(
            layer_idx, layers, elcr, target_idx
        )
        # Lower layer connections
        if layer_idx < depth - 1:
            llcr = random.randint(*self.llcr)
            self.extend_target_idx(
                layer_idx + 1, layers, llcr, target_idx
            )
        organism.connect_cells(cell.id, target_idx)

    def extend_target_idx(self, layer_idx, layers, max_connectivity, target_idx):
        target_idx.extend(
            self.get_target_idx(
                layers, layer_idx, max_connectivity
            )
        )

    def get_target_idx(self, layers: List[Layer],
                       target_layer_idx: int, max_connectivity):
        target_layer = layers[target_layer_idx]
        if target_layer:
            connectivity = min(max_connectivity, len(target_layer.children))
            target_cells = random.sample(
                target_layer.children, connectivity
            )
            target_ids = [cell.id for cell in target_cells]
            return target_ids
        return []
