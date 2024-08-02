# organism.py
import sys

import numpy as np

from core.cell.cell import Cell
from core.cell.collections.builtin_functors import Linker
from core.cell.geoo import GeneExpressedOptimizableOperand
from core.cell.optim.optimizable import MultiTree, multi_tree_derive
from core.cell.optim.optimization_args import OptimizationArgs
from core.cell.optim.optimizer import Optimizer, Optimization
from core.organism.node import Node


class Organism(Cell):
    def __init__(self, root, dim_in, depth=2, optimizer=None):
        super().__init__(root, 1, depth, optimizer)
        self.optimizer = optimizer
        self.weight_cache = None
        self.dim_in = dim_in

    def derive_unchained(self, index, by_weights=True):
        pass

    def get_weights_local(self):
        if self.weight_cache is None:
            self.weight_cache = []
            for node in self.children:
                self.weight_cache.extend(node.get_weights_local())
        return self.weight_cache

    def __call__(self, args, meta_args=None):
        return self.root(args, meta_args)

    def clone(self) -> "Organism":
        pass

    def to_python(self) -> str:
        pass

    def nodes(self):
        self.root.get_children()

    def link(self, next_chain):
        self.root = Linker(next_chain, self.root, self.dim_in)
        self.clean()

    def replace(self, node_old, node_new):
        pass

    def clean(self):
        self.derivative_cache.clear()
        if self.weight_cache is not None:
            self.weight_cache.clear()
            self.weight_cache = None

    def randomly_replace(self, mutant_node):
        pass

    @staticmethod
    def create_simple_organism(dim_in, dim_hidden, hidden_count, dim_out,
                               activation_function, spread_point=-1,
                               optimizer=None):
        input_node = Node(1, dim_in, dim_hidden, activation_function)
        optimizer = optimizer or OrganismOptimizer()
        organism = Organism(input_node, dim_in, hidden_count + 2,
                            optimizer)
        if spread_point == -1:
            spread_point = hidden_count + 1
        for i in range(1, hidden_count + 1):
            hidden_node = Node(1, dim_hidden, dim_hidden,
                               activation_function)
            organism.link(hidden_node)

        output_node = Node(1, dim_hidden, dim_out,
                           activation_function)
        organism.link(output_node)
        print("KWAAA")
        return organism

    def optimize(self, args: OptimizationArgs):
        print("DSDSSDSD")
        self.optimizer(self, args)


class OrganismOptimization(Optimization):
    def calculate_gradients(self):
        return self.fit_func.multi_gradient(self.cell, self.input,
                                            self.desired_output, self.weights)


class OrganismOptimizer(Optimizer):
    def make_optimizer(self, cell, optim_args, ewc_lambda=0.0,
                       l2_lambda=0.0):
        optim_args = optim_args.clone()
        optimizer = OrganismOptimization(cell, optim_args, self.risk,
                                         ewc_lambda=ewc_lambda,
                                         l2_lambda=l2_lambda)
        return optimizer

    def train(self, model, optimization_args):
        optimizer = self.make_optimizer(model, optimization_args)
        optimizer.optimize()

    def __call__(self, model, optimization_args):
        a = optimization_args
        print("ASASAS")
        for epoch in range(a.epochs):
            epoch_loss = 0
            for X_batch, y_batch in a.batches():
                input_data = [[np.array(x)] for x in X_batch]
                desired_output = [[np.array([y[0]])] for y in y_batch]

                optimization_args = OptimizationArgs(
                    inputs=input_data,
                    desired_output=desired_output,
                    loss_function=a.loss_function,
                    learning_rate=a.learning_rate,
                    max_iter=a.max_iter,  # Iteration within a batch
                    min_error=sys.maxsize
                )
                self.train(model, optimization_args)
                # epoch_loss += self.train(model, optimization_args)
