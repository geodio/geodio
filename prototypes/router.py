from typing import List, Optional, Dict

import numpy as np

from core.cell.cell import HALLOW_CELL
from core.cell.geoo import GeneExpressedOptimizableOperand, t_geoo
from core.cell.operands.constant import ONE
from core.cell.operands.weight import Weight, t_weight
from core.cell.train.optimization_args import OptimizationArgs
from core.cell.train.optimizer import FisherOptimizer


def rand_weight() -> Weight:
    return Weight(np.random.rand(), adaptive_shape=True)


class Router(GeneExpressedOptimizableOperand):
    def __init__(self,
                 nodes: List[t_geoo] = None,
                 weights: Optional[Dict[int, t_weight]] = None,
                 optimizer=None):
        optimizer = optimizer or FisherOptimizer()
        super().__init__(0, 2, optimizer=optimizer)
        self.children = [] if nodes is None else nodes
        self.weights = weights or self._initialize_weights()
        self._validate_weights()

    def _initialize_weights(self):
        return {i: rand_weight() for i in range(len(self.children))}

    def _validate_weights(self):
        # Ensure weights correspond to the number of children
        num_children = len(self.children)
        for i in range(num_children):
            if i not in self.weights:
                self.weights[i] = rand_weight()

        # Remove any extra weights
        self.weights = {i: self.weights[i] for i in range(num_children)}

    def nodes(self) -> List[t_geoo]:
        return self.children

    def replace(self, node_old: t_geoo, node_new: t_geoo):
        for i, node in enumerate(self.children):
            if node == node_old:
                self.children[i] = node_new
                break
        self._validate_weights()

    def add_node(self, node: t_geoo, weight: t_weight):
        node_id = len(self.children)
        self.children.append(node)
        self.weights[node_id] = weight

    def clean(self):
        self.children = []
        self.weights = {}

    def randomly_replace(self, mutant_node: t_geoo):
        if self.children:
            index = np.random.randint(len(self.children))
            self.children[index] = mutant_node
        self._validate_weights()

    def __call__(self, args, meta_args=None):
        weighted_sum = sum(
            node.get_state_weight()(args) * self.weights[i](args)
            for i, node in enumerate(self.children)
        )
        return weighted_sum

    def __invert__(self):
        return self.clone()

    def clone(self) -> "Router":
        return Router(self.children[:], self.weights.copy(),
                      self.optimizer)

    def to_python(self) -> str:
        weights = [self.weights[i] for i in range(len(self.children))]
        to_python = " + ".join([
            child.get_state_weight().to_python() + " * " + weight.to_python()
            for child, weight in zip(self.children, weights)
        ])
        return to_python

    def derive_uncached(self, index, by_weights=True):
        if not by_weights:
            return self
        inner_index = -1
        for i, node in enumerate(self.children):
            if node.get_state_weight().w_index == index:
                inner_index = i
        if inner_index == -1:
            for i, weight in self.weights.items():
                if weight.w_index == index:
                    inner_index = i
            if inner_index == -1:
                return self
            else:
                clone = self.clone()
                clone.weights[inner_index] = ONE
                return clone
        else:
            clone = self.clone()
            clone.children[inner_index] = HALLOW_CELL
            return clone

    def optimize(self, args: OptimizationArgs):
        self.optimizer(self, args)
        self._validate_weights()

    def get_weights_local(self):
        weights = [node.get_state_weight() for node in self.children]
        weights.extend(self.weights.values())
        return weights

    def set_weights(self, new_weights):
        old_weights = self.get_weights()
        for old_w, new_w in zip(old_weights, new_weights):
            new_w.set(old_w)

    def mutate(self, generator, max_depth=None):
        # TODO
        pass

    def mark_checkpoint(self):
        for node in self.children:
            node.mark_checkpoint()

    def get_gradients(self):
        pass

    def forward(self, x, meta_args=None):
        pass

