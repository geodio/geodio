import numpy as np

from src.cell.cell import Cell, t_cell_list
from src.cell.collections.builtin_functors import Prod, Add
from src.cell.operands.weight import Weight
from src.cell.optim.fitness import FitnessFunction


class LinkException(Exception):
    def __init__(self, link_size, cell_arity, weight_count):
        message = (f"Arity of the internal cell must be 1. "
                   f"Link size must be equal to weight count. "
                   f"Link size: {link_size}; "
                   f"Cell arity: {cell_arity}; "
                   f"Weight count: {weight_count}; ")
        super().__init__(message)


def new_a_w(w=0.0):
    return Weight(w, adaptive_shape=True)


class Link(Cell):
    def __init__(self, linked_cells: t_cell_list, internal_cell: Cell,
                 weights, max_depth):
        self.link_size = len(linked_cells)
        if internal_cell.arity != 1:
            raise LinkException(self.link_size,
                                internal_cell.arity,
                                len(weights))

        self.linked_cells = linked_cells
        self.internal_cell = internal_cell
        self.weights = w = weights

        nuclei = [
            Prod([new_a_w(w[i]), self.linked_cells[i]])
            for i in range(self.link_size)
        ]

        root = Add(nuclei, self.link_size)

        super().__init__(root, 0, max_depth)
        self.internal_cell = internal_cell  # The internal cell

    def __call__(self, args):
        # Calculate weighted sum of inputs
        weighted_sum = sum(w * arg for w, arg in zip(self.weights, args))
        return self.internal_cell([weighted_sum])

    def derive(self, var_index, by_weights=True):

        derived_links = [
            cell.derive(var_index, by_weights) for cell in self.linked_cells
        ]
        root_weights = [weight.weight for weight in self.root.get_weights()]
        derived_link = Link(derived_links, self.internal_cell,
                            root_weights, self.depth)

        self.internal_cell.derive(var_index, by_weights)

    def read_linked_states(self):
        return [cell.size for cell in self.linked_cells]

    def optimize_values(
            self, fit_fct: FitnessFunction,
            variables, desired_output, learning_rate=0.1,
            max_iterations=100, min_fitness=10
    ):
        self.internal_cell.optimize_values(
            fit_fct, variables, desired_output,
            learning_rate, max_iterations, min_fitness
        )
        return self.internal_cell.get_weights()

    def get_weights(self):
        linked = [self.root, self.internal_cell]
        weights = []

        for child in linked:
            weights.extend(child.get_weights())

        for i, weight in enumerate(weights):
            weight.w_index = i

        return weights

    def set_weights(self, new_weights):
        self.weights = new_weights
        self.internal_cell.set_weights(new_weights)

    def __repr__(self):
        return (f"Link with weights: {self.weights},"
                f" internal cell: {repr(self.internal_cell)}")

    def __str__(self):
        return (f"Link: weights = {self.weights},"
                f" internal cell = {str(self.internal_cell)}")

    @staticmethod
    def from_root(root, internal_cell, l_size, max_depth) -> 'Link':
        weights = np.zeros(l_size)
        nuclei = [internal_cell for _ in range(l_size)]
        link = Link(nuclei, internal_cell, weights, max_depth)
        link.root = root
        return link
