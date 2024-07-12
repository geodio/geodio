import sys
from enum import IntEnum
from typing import List, Optional

import numpy as np

from core.cell.cell import Cell
from core.cell.geoo import t_geoo
from core.cell.operands.stateful import Stateful
from core.cell.optim.optimizable import OptimizableOperand
from core.cell.optim.optimization_args import OptimizationArgs
from core.cell.optim.optimizer import FisherOptimizer
from core.organism.link import Link


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


class Layer(OptimizableOperand, Stateful):
    def __init__(self, arity, cells: Optional[List[t_geoo]] = None,
                 distribution_policy=DistributionPolicy.IGNORE,
                 layer_type=LayerType.HIDDEN, optimizer=None):
        if optimizer is None:
            optimizer = FisherOptimizer()

        OptimizableOperand.__init__(self, arity, optimizer)
        Stateful.__init__(self)

        self.distribution_policy = distribution_policy
        self.children: List[t_geoo] = cells if cells is not None else []
        self.fitness = sys.maxsize
        self.layer_type = layer_type
        self.optimizer.risk = True
        self.id = np.random.randint(1, 10000)

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
            return link

    def __invert__(self):
        pass

    def clone(self) -> "Layer":
        return Layer(arity=self.arity,
                     cells=[cell.clone() for cell in self.children],
                     distribution_policy=self.distribution_policy)

    def to_python(self) -> str:
        return f"layer([{', '.join([str(child) for child in self.children])}])"

    def derive(self, index, by_weights=True):
        # Correctly aggregate the derivatives of the cells
        derived_cells = [child.derive(index, by_weights) for child in
                         self.children]
        return Layer(self.arity, derived_cells, self.distribution_policy,
                     self.layer_type, self.optimizer)

    def call_according_to_policy(self, args):
        if self.distribution_policy == DistributionPolicy.UNIFORM:
            return [cell(args) for cell in self.children]
        if self.distribution_policy == DistributionPolicy.IGNORE:
            return [cell(args) for cell in self.children]
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

    def optimize(self, args: OptimizationArgs):
        if self.layer_type == LayerType.OUTPUT:
            args = args.clone()
            if len(args.inputs) > 1:
                args.inputs = [[["O"]], [["O"]]]
            else:
                args.inputs = [[["O"]]]
        elif self.layer_type == LayerType.INPUT:
            args = args.clone()
            self._handle_input_optimization(args)
            return
        else:
            args = args.clone()
            if len(args.inputs) > 1:
                args.inputs = [[["H"]], [["H"]]]
                args.desired_output = [link.state for link in self.children]
                args.desired_output = [
                    link.checkpoint for link in self.children
                ]
            else:
                args.inputs = [[["H"]]]
                args.desired_output = [link.state for link in self.children]
        self.optimizer(self, args)

    def _handle_input_optimization(self, args: OptimizationArgs):
        def optimize_cell(input_cell, input_vars):
            args_clone = args.clone()
            args_clone.desired_output = [input_cell.state]
            args_clone.inputs = input_vars
            input_cell.optimize(args_clone)

        if self.distribution_policy == DistributionPolicy.UNIFORM:
            for cell in self.children:
                optimize_cell(cell, args.inputs)
        elif self.distribution_policy == DistributionPolicy.SPLIT:
            split_inputs = self.split_args(args.inputs)
            for cell, inputs in zip(self.children, split_inputs):
                optimize_cell(cell, [inputs])
        else:
            for cell in self.children:
                optimize_cell(cell, [])

    def get_weights(self):
        weights = []
        seen = set()
        for child in self.children:
            for weight in child.get_weights():
                if weight not in seen:
                    seen.add(weight)
                    weights.append(weight)
        return weights

    def set_weights(self, new_weights):
        offset = 0
        seen = set()
        for child in self.children:
            child_weights = child.get_weights()
            num_weights = len(child_weights)
            if num_weights > 0:
                for i, weight in enumerate(child_weights):
                    if weight not in seen:
                        seen.add(weight)
                        child.set_weights(
                            new_weights[offset:offset + num_weights])
                        offset += num_weights

    def mark_checkpoint(self):
        for cell in self.children:
            cell.mark_checkpoint()
