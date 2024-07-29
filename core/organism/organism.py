# organism.py
from typing import List

import numpy as np

from core.organism.backpropagation import Backpropagatable, BPOptimizer


class Organism(Backpropagatable):
    def __init__(self, nodes):
        super().__init__(1, BPOptimizer())
        self.weight_cache = None
        self.children = nodes

    def backpropagation(self, dx: np.ndarray) -> np.ndarray:
        for node in reversed(self.children):
            dx = [node.backpropagation(dx)]
        return dx

    def get_gradients(self) -> List[np.ndarray]:
        gradients = []
        for node in self.children:
            gradients.extend(node.get_gradients())
        return gradients

    def derive_unchained(self, index, by_weights):
        pass

    def get_weights_local(self):
        if self.weight_cache is None:
            self.weight_cache = []
            for node in self.children:
                self.weight_cache.extend(node.get_weights_local())
        return self.weight_cache

    def __call__(self, args):
        x = args[0]
        for node in self.children:
            x = node([x])
        return x

    def clone(self) -> "Organism":
        pass

    def to_python(self) -> str:
        pass
