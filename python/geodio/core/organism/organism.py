# organism.py

from geodio.core.cell import EpochedOptimizer, OptimizableOperand, Backpropagatable, \
    LinearTransformation, b_var
from geodio.core.cell import Linker
from geodio.core.cell import ActivationFunction

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
            for child in self.get_sub_operands():
                weights.extend(child.get_weights_local())
            self.weight_cache = weights
        return self.weight_cache

    def __call__(self, args, meta_args=None):
        x = args[0]
        return self.forward(x, meta_args)

    def forward(self, x, meta_args=None):
        for child in self.get_sub_operands():
            x = child.forward(x, meta_args)
        return x

    def backpropagation(self, dx, meta_args=None):
        for child in self.get_sub_operands()[::-1]:
            dx = child.backpropagation(dx, meta_args)
        return dx

    def clone(self) -> "Organism":
        pass

    def get_local_gradients(self) -> list:
        pass

    def to_python(self) -> str:
        pass

    def nodes(self):
        self.get_sub_operands()

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
                               activation_function: ActivationFunction,
                               spread_point=-1,
                               optimizer=None):
        # TODO THIS HAS NOTHING TO DO WITH ORGANISM
        organism = activation_function.clone()
        organism.optimizer = optimizer or EpochedOptimizer()
        dummy = b_var()

        class Link:
            # TODO THIS IS A DEMO FOR THE BLUEPRINT GENERATION
            # WHERE THE BLUEPRINT IS OBTAINED FROM THE GENES

            def __init__(self, org):
                self.last_operand = org

            def __call__(self, next_op):
                self.last_operand.set_child(next_op)
                self.last_operand = next_op

        link = Link(organism)
        link(LinearTransformation(dim_hidden, dim_out, [dummy]))

        for i in range(1, hidden_count + 1):
            link(activation_function.clone())
            link(LinearTransformation(dim_hidden, dim_hidden, [dummy]))

        link(activation_function.clone())
        link(LinearTransformation(dim_in, dim_hidden, [dummy]))

        organism.set_optimization_risk(True)
        return organism
