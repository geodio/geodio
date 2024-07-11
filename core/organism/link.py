import random
import sys
from typing import Union

import numpy as np

from core.cell.cell import Cell
from core.cell.collections.builtin_functors import Prod, Add
from core.cell.operands.weight import Weight
from core.cell.optim.loss import LossFunction


class LinkException(Exception):
    def __init__(self, link_size, cell_arity, weight_count):
        message = (f"Arity of the internal cell must be 1. "
                   f"Link size must be equal to weight count. "
                   f"Link size: {link_size}; "
                   f"Cell arity: {cell_arity}; "
                   f"Weight count: {weight_count}; ")
        super().__init__(message)


def new_link_weight(w: Union[Weight, float, np.ndarray] = None):
    if w is None:
        w = random.random()
    if isinstance(w, Weight):
        return w
    return Weight(w, adaptive_shape=True)


def build_link_root(internal_cell, linked_cells, weights=None):
    if weights is None:
        weights = [
            new_link_weight() for _ in range(len(linked_cells))
        ]
    link_size = len(linked_cells)
    if internal_cell.arity != 1:
        raise LinkException(link_size,
                            internal_cell.arity,
                            len(weights))
    w = weights
    nuclei = [
        Prod([
            new_link_weight(w[i]), linked_cells[i].get_state_weight()
        ]) for i in range(link_size)
    ]
    root = Add(nuclei, link_size)
    return root


class Link(Cell):
    def __init__(self, root, internal_cell: Cell, max_depth):
        super().__init__(root, 0, max_depth)
        self.internal_cell = internal_cell
        self.error = sys.maxsize
        self.id = self.internal_cell.id
        self.set_optimization_risk(True)

    def __call__(self, args):
        # Calculate weighted sum of inputs
        hidden_cell_input = [self.root(args)]
        hidden_cell_output = self.internal_cell(hidden_cell_input)
        return hidden_cell_output

    def derive(self, var_index, by_weights=True):
        derived_root = self.root.derive(var_index, by_weights)
        derived_cell = self.internal_cell.derive(var_index, by_weights)
        derived_cell_linked = Link(self.root, derived_cell, self.depth)
        derived_by_input_cell = self.internal_cell.derive(0, False)
        derived_link = Link(self.root, derived_by_input_cell, self.depth)
        derivative = Add([
            Prod([
                derived_root,
                derived_link,
            ]),
            derived_cell_linked
        ], 2)
        return derivative

    def get_weights(self):
        linked = [self.internal_cell, self.root]
        weights = []

        for child in linked:
            weights.extend(child.get_weights())

        for i, weight in enumerate(weights):
            weight.w_index = i
        return weights

    def set_weights(self, new_weights):
        linked = [self.internal_cell, self.root]
        offset = 0
        for child in linked:
            child_weights = child.get_weights()
            num_weights = len(child_weights)
            if num_weights > 0:
                child.set_weights(new_weights[offset:offset + num_weights])
                offset += num_weights

    @staticmethod
    def from_cells(cell, linked_cells, weights=None):
        return Link(build_link_root(cell, linked_cells, weights),
                    cell, 2)

    def optimize_values(self, fit_fct: LossFunction, variables,
                        desired_output,
                        learning_rate=0.1,
                        max_iterations=100,
                        min_error=10):
        if desired_output is None:
            desired_output = self.state
        if variables is None:
            # To have input of same length as input.
            variables = desired_output
        self.optimizer(self, desired_output, fit_fct, learning_rate,
                       max_iterations, variables)

    def to_python(self):
        return (f"{self.root.to_python()} -> {self.internal_cell.id} => "
                f"{self.internal_cell.to_python()}")
