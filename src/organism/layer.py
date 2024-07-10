import sys
from enum import IntEnum
from typing import List

from src.cell.cell import Cell
from src.organism.link import Link
from src.cell.operands.operand import Operand
from src.cell.optim.fitness import FitnessFunction
from src.cell.optim.optimizable import Optimizable


class DistributionPolicy(IntEnum):
    SPLIT = 0
    """The input is distributed to the cells in the layer."""

    UNIFORM = 1
    """All cells in the layer receive the same input."""

    IGNORE = 3
    """
    Cells in the layer receive empty input.
    Any provided arguments are ignored.
    """


def split_distribution_msg(arity: int, len_args: int) -> str:
    msg = (
        f"Layer with Split Distribution Policy has cumulative cell arity = "
        f"{arity}, yet {len_args} were provided. The rest of the arguments "
        f"will be {'ignored' if len_args > arity else '0'}."
    )
    return msg


class DistributionWarning(Warning):
    def __init__(self, msg):
        super().__init__(msg)


class LayerType(IntEnum):
    INPUT = 0
    OUTPUT = 1
    HIDDEN = 2


class Layer(Operand, Optimizable):
    def __init__(self, arity, cells=None,
                 distribution_policy=DistributionPolicy.IGNORE,
                 layer_type=LayerType.HIDDEN, ):
        super().__init__(arity)
        self.distribution_policy = distribution_policy
        self.children = cells if cells is not None else []
        self.fitness = sys.maxsize
        self.layer_type = layer_type

    def add_cell(self, cell: Cell):
        self.children.append(cell)

    def add_cells(self, cells: List[Cell]):
        self.children.extend(cells)

    def __call__(self, args):
        outputs = self.call_according_to_policy(args)
        self.update_cells_states(outputs)
        return outputs

    def update_cells_states(self, new_states):
        cell_count = len(self.children)
        assert cell_count == len(new_states), ("Failed to update cell "
                                               "states, number of "
                                               "cells is different "
                                               "than number of "
                                               "states provided:"
                                               f"[no. cells = "
                                               f"{cell_count}],"
                                               f"[no. states = "
                                               f"{len(new_states)}]")
        for i in range(len(new_states)):
            self.children[i].state = new_states[i]

    def get_cell_states(self):
        return [cell.state for cell in self.children]

    def link(self, cell: Cell,
             linked_cells: List[Cell],
             initial_weights=None):
        if cell in self.children:
            index = self.children.index(cell)
            self.children.remove(cell)
            link = Link.from_cells(cell, linked_cells, initial_weights)
            self.children.insert(index, link)

    def __invert__(self):
        pass

    def clone(self) -> "Layer":
        return Layer(arity=self.arity,
                     cells=[cell.clone() for cell in self.children],
                     distribution_policy=self.distribution_policy)

    def to_python(self) -> str:
        return f"layer([{', '.join([str(child) for child in self.children])}])"

    def derive(self, index, by_weights=True):
        pass

    def call_according_to_policy(self, args):
        if self.distribution_policy == DistributionPolicy.UNIFORM:
            return [cell(args) for cell in self.children]
        if self.distribution_policy == DistributionPolicy.IGNORE:
            empty_list = []
            return [cell(empty_list) for cell in self.children]
        if self.distribution_policy == DistributionPolicy.SPLIT:
            return self._split_distribution_call(args)

    def _split_distribution_call(self, args):
        cell_args_list = self.split_args(args, True)

        return [
            cell(cell_args)
            for cell, cell_args in zip(self.children, cell_args_list)
        ]

    def split_args(self, args, raise_warning=False):
        arity_sum = sum([
            cell.arity for cell in self.children
        ])
        if raise_warning and len(args) != arity_sum:
            raise DistributionWarning(
                split_distribution_msg(arity_sum, len(args))
            )
        offset = 0
        cell_args_list = []
        left = arity_sum - len(args)
        if left > 0:
            args.extend([0 for _ in range(left)])
        for cell in self.children:
            cell_args_list.append(args[offset:offset + cell.arity])
            offset += cell.arity
        return cell_args_list

    def __repr__(self):
        x = ",".join([str(cell.id) for cell in self.children])
        return f"<layer[{x}]>"

    def optimize_values(self, fit_fct: FitnessFunction, variables,
                        desired_output,
                        learning_rate=0.1,
                        max_iterations=100,
                        min_fitness=sys.maxsize):
        if self.layer_type == LayerType.OUTPUT:
            for link, output in zip(self.children, desired_output):
                link.state = output
                link.optimize_values(fit_fct, None, [link.state],
                                     learning_rate,
                                     max_iterations, min_fitness)
        elif self.layer_type == LayerType.INPUT:
            self._handle_input_optimization(variables, fit_fct,
                                            learning_rate, max_iterations,
                                            min_fitness)
        else:
            for link in self.children:
                link.optimize_values(fit_fct, None, [link.state],
                                     learning_rate,
                                     max_iterations, min_fitness)

    def _handle_input_optimization(
            self, variables, fit_fct, learning_rate, max_iterations,
            min_fitness):
        def optimize_cell(input_cell, input_vars):
            input_cell.optimize_values(
                fit_fct, input_vars, [input_cell.state],
                learning_rate, max_iterations, min_fitness
            )

        if self.distribution_policy == DistributionPolicy.UNIFORM:
            for cell in self.children:
                optimize_cell(cell, variables)
        elif self.distribution_policy == DistributionPolicy.SPLIT:
            split_inputs = self.split_args(variables)
            for cell, inputs in zip(self.children, split_inputs):
                optimize_cell(cell, [inputs])
        else:
            for cell in self.children:
                optimize_cell(cell, [])
