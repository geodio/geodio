import random
import sys
from typing import Union

import numpy as np

from src.cell.cell import Cell, t_cell_list
from src.cell.collections.builtin_functors import Prod, Add
from src.cell.operands.constant import ONE, ZERO
from src.cell.operands.operand import Operand
from src.cell.operands.weight import Weight, AbsWeight
from src.cell.optim.fitness import FitnessFunction


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


class State(AbsWeight):

    def __init__(self, cell: Cell, adaptive_shape=False):
        super().__init__(adaptive_shape)
        self.cell = cell

    def d(self, var_index):
        if isinstance(self.cell.state, np.ndarray):
            return Weight(np.zeros_like(self.cell.state))
        return ZERO

    def d_w(self, dw):
        state = self.cell.state
        if isinstance(state, (np.ndarray, list)):
            return Weight(np.ones_like(state)) \
                if self.w_index == dw \
                else Weight(np.zeros_like(state))
        return ONE if self.w_index == dw else ZERO

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def clone(self):
        cloned_cell = self.cell.clone()
        w_clone = State(cloned_cell, self.adaptive_shape)
        w_clone.w_index = self.w_index
        return w_clone

    def to_python(self) -> str:
        return f"_state({str(self.cell.state)})"

    def set(self, weight: Union[np.ndarray, float]) -> None:
        if isinstance(weight, AbsWeight):
            self.cell.state = weight.get()
        else:
            self.cell.state = weight

    def get(self) -> Union[np.ndarray, float]:
        return self.cell.state


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
            new_link_weight(w[i]), State(linked_cells[i])
        ]) for i in range(link_size)
    ]
    root = Add(nuclei, link_size)
    return root


class Link(Cell):
    def __init__(self, root, internal_cell: Cell, max_depth):
        super().__init__(root, 0, max_depth)
        self.internal_cell = internal_cell
        self.fitness = sys.maxsize

    def __call__(self, args):
        # Calculate weighted sum of inputs
        hidden_cell_input = [self.root(args)]
        hidden_cell_output = self.internal_cell(hidden_cell_input)
        return hidden_cell_output

    def derive(self, var_index, by_weights=True):
        derived_root = self.root.derive(var_index, by_weights)
        derived_cell = self.internal_cell.derive(var_index, by_weights)
        derived_by_input_cell = self.internal_cell.derive(0, False)
        derivative = Add([
            Prod([
                derived_root,
                Link(self.root, derived_by_input_cell, self.depth),
            ]),
            derived_cell
        ], 2)
        return derivative

    def get_weights(self):
        linked = [self.root, self.internal_cell]
        weights = []

        for child in linked:
            weights.extend(child.get_weights())

        for i, weight in enumerate(weights):
            weight.w_index = i

        return weights

    def set_weights(self, new_weights):
        linked = [self.root, self.internal_cell]
        offset = 0
        for child in linked:
            child_weights = child.get_weights()
            num_weights = len(child_weights)
            if num_weights > 0:
                child.set_weights(new_weights[offset:offset + num_weights])
                offset += num_weights

    def __repr__(self):
        return (f"Link with root: {self.root},"
                f" internal cell: {repr(self.internal_cell)}")

    def __str__(self):
        return (f"Link: root = {self.root},"
                f" internal cell = {str(self.internal_cell)}")

    @staticmethod
    def from_cells(cell, linked_cells, weights=None):
        return Link(build_link_root(cell, linked_cells, weights),
                    cell, 2)

    def optimize_values(self, fit_fct: FitnessFunction, variables,
                        desired_output,
                        learning_rate=0.1,
                        max_iterations=100,
                        min_fitness=10):
        if desired_output is None:
            desired_output = self.state
        self.optimizer(self, desired_output, fit_fct, learning_rate,
                       max_iterations, variables)
