# layer.py
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
    UNIFORM = 1
    IGNORE = 2


class LayerType(IntEnum):
    INPUT = 0
    OUTPUT = 1
    HIDDEN = 2


class Layer(OptimizableOperand, Stateful):
    def __init__(self, arity, cells: Optional[List[t_geoo]] = None,
                 distribution_policy=DistributionPolicy.IGNORE,
                 layer_type=LayerType.HIDDEN, optimizer=None):
        super().__init__(arity, optimizer if optimizer else FisherOptimizer())
        self.children: List[t_geoo] = cells if cells is not None else []
        self.distribution_policy = distribution_policy
        self.layer_type = layer_type
        self.id = np.random.randint(1, 10000)

    def add_cell(self, cell: Cell):
        self.children.append(cell)

    def add_cells(self, cells: List[Cell]):
        self.children.extend(cells)

    def __call__(self, args, meta_args=None):
        outputs = self.call_according_to_policy(args)
        self.update_cells_states(outputs)
        return outputs

    def call_according_to_policy(self, args):
        if self.distribution_policy in (DistributionPolicy.UNIFORM,
                                        DistributionPolicy.IGNORE):
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
        arity_sum = sum(cell.arity for cell in self.children)
        if raise_warning and len(args) != arity_sum:
            msg = (f"Layer with Split Distribution Policy has cumulative "
                   f"cell arity = {arity_sum}, yet {len(args)} were "
                   f"provided. The rest of the arguments will be "
                   f"{'ignored' if len(args) > arity_sum else '0'}.")
            raise Warning(msg)
        offset = 0
        cell_args_list = []
        left = arity_sum - len(args)
        args.extend([0] * left) if left > 0 else None
        for cell in self.children:
            cell_args_list.append(args[offset:offset + cell.arity])
            offset += cell.arity
        return cell_args_list

    def update_cells_states(self, new_states):
        assert len(self.children) == len(new_states), ("Mismatch in number of "
                                                       "cells and states.")
        for i in range(len(new_states)):
            self.children[i].state = new_states[i]

    def get_cell_states(self):
        return [cell.state for cell in self.children]

    def link(self, cell: Cell, linked_cells: List[Cell], initial_weights=None):
        if cell in self.children:
            index = self.children.index(cell)
            self.children.remove(cell)
            link = Link.from_cells(cell, linked_cells, initial_weights)
            self.children.insert(index, link)
            return link

    def clone(self) -> "Layer":
        return Layer(self.arity, [cell.clone() for cell in self.children],
                     self.distribution_policy, self.layer_type, self.optimizer)

    def to_python(self) -> str:
        return f"layer([{', '.join([str(child) for child in self.children])}])"

    def derive(self, index, by_weights=True):
        derived_cells = [child.derive(index, by_weights) for child in
                         self.children]
        return Layer(self.arity, derived_cells, self.distribution_policy,
                     self.layer_type, self.optimizer)

    def optimize(self, args: OptimizationArgs):
        if self.layer_type == LayerType.INPUT:
            self._handle_input_optimization(args)
            return
        cloned_args = args.clone()
        if self.layer_type == LayerType.OUTPUT:
            cloned_args.inputs = [[["O"]]] if len(args.inputs) <= 1 else [
                [["O"]], [["O"]]]
        elif self.layer_type == LayerType.HIDDEN:
            cloned_args.inputs = [[["H"]]] if len(args.inputs) <= 1 else [
                [["H"]], [["H"]]]
            cloned_args.desired_output = [link.state for link in self.children]
            cloned_args.desired_output.extend(
                [link.checkpoint for link in self.children])
        self.optimizer(self, cloned_args)

    def _handle_input_optimization(self, args: OptimizationArgs):
        def optimize_cell(cell, inputs):
            cell_args = args.clone()
            cell_args.desired_output = [cell.state]
            cell_args.inputs = inputs
            cell.optimize(cell_args)

        if self.distribution_policy == DistributionPolicy.UNIFORM:
            for cell in self.children:
                optimize_cell(cell, args.inputs)
        elif self.distribution_policy == DistributionPolicy.SPLIT:
            cell_args_list = self.split_args(args.inputs, raise_warning=True)
            for cell, cell_args in zip(self.children, cell_args_list):
                optimize_cell(cell, cell_args)
