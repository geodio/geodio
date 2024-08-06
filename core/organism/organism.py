# organism.py
import sys

from core import logger
from core.cell import BackpropagatableOperand, EpochedOptimizer
from core.cell import Linker
from core.cell import OptimizationArgs
from core.cell import Optimizer, Optimization
from core.organism.node import Node


class Organism(BackpropagatableOperand):
    def __init__(self, children, dim_in, arity, optimizer=None):
        super().__init__(arity, optimizer)
        self.weight_cache = None
        self.dim_in = dim_in
        self.children = children
        self.root = None

    def derive_uncached(self, index, by_weights=True):
        if self.root is None:
            self.root = self.children[0]
            for i in range(1, len(self.children)):
                self.root = Linker(self.children[i], self.root)
        return self.root.derive_uncached(index, by_weights)

    def get_weights_local(self):
        if self.weight_cache is None:
            weights = []
            for child in self.get_children():
                weights.extend(child.get_weights_local())
            self.weight_cache = weights
        return self.weight_cache

    def forward(self, x, meta_args=None):
        args = [x]
        for child in self.get_children():
            args = [child(args, meta_args)]
        return args[0]

    def backpropagation(self, dx, meta_args=None):
        for child in self.get_children()[::-1]:
            dx = child.backpropagation(dx, meta_args)
        return dx

    def clone(self) -> "Organism":
        pass

    def to_python(self) -> str:
        pass

    def get_gradients(self):
        gradients = []
        for child in self.get_children():
            gradients.extend(child.get_gradients())
        return gradients

    def nodes(self):
        self.get_children()

    def link(self, next_chain):
        self.children.append(next_chain)
        self.clean()

    def clean(self):
        self.derivative_cache.clear()
        if self.weight_cache is not None:
            self.weight_cache.clear()
            self.weight_cache = None
        self.root = None

    @staticmethod
    def create_simple_organism(dim_in, dim_hidden, hidden_count, dim_out,
                               activation_function, spread_point=-1,
                               optimizer=None, backprop=False):
        input_node = Node(1, dim_in, dim_hidden, activation_function.clone())
        optimizer = optimizer or EpochedOptimizer(backprop)
        if spread_point == -1:
            spread_point = hidden_count + 1
        children = [input_node]
        for i in range(1, hidden_count + 1):
            hidden_node = Node(1, dim_hidden, dim_hidden,
                               activation_function.clone())
            children.append(hidden_node)

        output_node = Node(1, dim_hidden, dim_out,
                           activation_function.clone())
        children.append(output_node)
        organism = Organism(children, dim_in, 1,
                            optimizer)
        organism.set_optimization_risk(True)
        return organism

    def optimize(self, args: OptimizationArgs):
        self.optimizer(self, args)
