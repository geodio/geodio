# organism.py

from core.cell import EpochedOptimizer, OptimizableOperand, Backpropagatable
from core.cell import Linker
from core.organism.node import Node


class Organism(OptimizableOperand, Backpropagatable):
    def __init__(self, children, dim_in, arity, optimizer=None):
        super().__init__(arity, optimizer)
        self.weight_cache = None
        self.dim_in = dim_in
        self.children = children
        self.root = None
        self.X = None

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

    def __call__(self, args, meta_args=None):
        x = args[0]
        return self.forward(x, meta_args)

    def forward(self, x, meta_args=None):
        for child in self.get_children():
            x = child.forward(x, meta_args)
        return x

    def backpropagation(self, dx, meta_args=None):
        for child in self.get_children()[::-1]:
            dx = child.backpropagation(dx, meta_args)
        return dx

    def clone(self) -> "Organism":
        pass

    def to_python(self) -> str:
        pass

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
                               optimizer=None):
        input_node = Node(1, dim_in, dim_hidden, activation_function.clone())
        optimizer = optimizer or EpochedOptimizer()
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

    def get_gradients(self):
        gradients = []
        for child in self.get_children():
            gradients.extend(child.get_gradients())
        return gradients
