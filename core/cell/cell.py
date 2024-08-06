from typing import TypeVar, List, Iterable

from core.cell.geoo import GeneExpressedOptimizableOperand
from core.cell.operands.constant import ONE
from core.cell.operands import Operand
from core.cell.optim.optimization_args import OptimizationArgs
from core.genetic.pop_utils import ReproductionPolicy
from core.cell.math import rnd


class Cell(GeneExpressedOptimizableOperand):
    def __init__(self, root: Operand, arity: int, max_depth,
                 reproduction_policy=ReproductionPolicy.DIVISION,
                 optimizer=None):
        super().__init__(arity, max_depth, reproduction_policy, optimizer)
        self.weight_cache = None
        self.root = root

    def nodes(self):
        return self.root.children

    def __call__(self, args, meta_args=None):
        if not isinstance(args, Iterable):
            args = [args]
        self.state = self.root(args, meta_args)
        return self.state

    def replace(self, node_old, node_new):
        self.root.replace_child(node_old, node_new)

    def to_python(self):
        return self.root.to_python()

    def randomly_replace(self, mutant_node):
        i = rnd.from_range(0, len(self.root.children), True)
        self.root.children[i] = mutant_node

    def optimize(self, args: OptimizationArgs):
        y_pred = [self(x_inst) for x_inst in args.inputs]
        if args.desired_output is None:
            return
        self.error = args.loss_function(args.desired_output, y_pred)
        if not (self.error <= args.min_error or self.marked):
            return
        args = args.clone()
        args.max_iter *= (1 / (self.age + 1))
        args.max_iter = int(args.max_iter)

        self.optimizer(self, args)
        return self.get_weights()

    def get_weights_local(self):
        if self.weight_cache is None:
            self.weight_cache = self.root.get_weights_local()
        return self.weight_cache

    def set_weights(self, new_weights):
        self.root.set_weights(new_weights)

    def derive_unchained(self, var_index, by_weights=True):
        derivative_root = self.root.derive(var_index, by_weights)
        derivative = Cell(derivative_root, self.arity, 0)
        return derivative

    def __invert__(self):
        return ~self.root

    def clone(self) -> "Cell":
        return Cell(self.root.clone(), self.arity, self.depth)

    def __repr__(self):
        return (f"root = {self.to_python()}, age = {self.age}, marked? "
                f"= {self.marked}, fitness = {self.error}")

    def __str__(self):
        return (f"Individual: {self.to_python()} \n"
                f"Error: {self.get_error()} \n"
                f"Age: {self.age} \n"
                f"Marked? {self.marked}\n"
                f"")

    def clean(self):
        self.error = None
        self.weight_cache = None
        self.derivative_cache = {}

    def __eq__(self, other):
        if not isinstance(other, Cell):
            return False
        return self.root == other.root


t_cell = TypeVar('t_cell', bound=Cell)
t_cell_list = TypeVar('t_cell_list', bound=List[Cell])
HALLOW_CELL = Cell(ONE, 0, 1)
HALLOW_CELL.state = 1.0
