from typing import TypeVar, List, Union

import numpy as np

from core.cell.operands.constant import ZERO, ONE
from core.cell.operands.weight import AbsWeight, Weight

from core.cell.optim.optimizable import Optimizable
from core.math import rnd
from core.cell.optim.fitness import FitnessFunction
from core.cell.operands.operand import Operand

from core.genetic.pop_utils import ReproductionPolicy
from core.cell.optim.optimizer import Optimizer


class State(AbsWeight):

    def __init__(self, cell: 'Cell', adaptive_shape=False):
        super().__init__(adaptive_shape)
        self.cell = cell
        self.cell.state_weight = self

    def d(self, var_index):
        if isinstance(self.cell.state, np.ndarray):
            return Weight(np.zeros_like(self.cell.state))
        return ZERO

    def d_w(self, dw):
        state = self.cell.state
        if isinstance(state, (np.ndarray, list)):
            return Weight(np.ones_like(state)) \
                if self.w_index == dw \
                else Weight(np.zeros_like(state))
        return ONE if self.w_index == dw else ZERO

    def derive(self, index, by_weights=True):
        if by_weights:
            return self.d_w(index)
        return self.d(index)

    def clone(self):
        cloned_cell = self.cell.clone()
        w_clone = State(cloned_cell, self.adaptive_shape)
        w_clone.w_index = self.w_index
        return w_clone

    def to_python(self) -> str:
        return f"_state({str(self.cell.state)} ~ {str(self.cell.id)})"

    def set(self, weight: Union[np.ndarray, float]) -> None:
        if isinstance(weight, AbsWeight):
            self.cell.state = weight.get()
        else:
            self.cell.state = weight

    def get(self) -> Union[np.ndarray, float]:
        return self.cell.state


class Cell(Operand, Optimizable):
    def __init__(self, root: Operand, arity: int, max_depth,
                 reproduction_policy=ReproductionPolicy.DIVISION,
                 optimizer=None):
        super().__init__(arity)
        self.weight_cache = None
        self.root = root
        self.depth = max_depth
        self.age = 0
        self.fitness = None
        self.marked = False
        self.reproduction_policy = reproduction_policy
        self.mutation_risk = 0.5
        self.derivative_cache = {}
        self.state = 0.0
        self.frozen = None
        if optimizer is None:
            self.optimizer = Optimizer()
        else:
            self.optimizer = optimizer
        self.state_weight = None

    def nodes(self):
        return self.root.children

    def __call__(self, args):
        return self.root(args)

    def replace(self, node_old, node_new):
        self.root.replace_child(node_old, node_new)

    def to_python(self):
        return self.root.to_python()

    def mutate(self, generator,
               max_depth=None):
        if not max_depth:
            max_depth = self.depth
        r = np.random.rand()
        if r < self.mutation_risk and len(self.root.children) != 0:
            mutant_node = generator.generate_node(max_depth - 1)
            self.depth = max_depth
            self._randomly_replace(mutant_node)
            self.fitness = None
            self.weight_cache = None
            self.derivative_cache = {}
        return self

    def _randomly_replace(self, mutant_node):
        i = rnd.from_range(0, len(self.root.children), True)
        self.root.children[i] = mutant_node

    def get_age(self):
        return self.age

    def inc_age(self, age_benefit=0):
        """
        ages the tree if the fitness exists
        :param age_benefit: contribution of age to the fitness
        :return:
        """
        if self.fitness is not None:
            self.age += 1
            self.fitness *= (1 - age_benefit)
            self.mutation_risk *= (1 - age_benefit)

    def mark(self):
        self.marked = True

    def optimize_values(self, fit_fct: FitnessFunction, variables,
                        desired_output,
                        learning_rate=0.1,
                        max_iterations=100,
                        min_fitness=10):
        y_pred = [self(x_inst) for x_inst in variables]
        if desired_output is None:
            return
        self.fitness = fit_fct(desired_output, y_pred)
        if not (self.fitness <= min_fitness or self.marked):
            return

        max_iterations *= (1 / (self.age + 1))
        max_iterations = int(max_iterations)

        self.optimizer(self, desired_output, fit_fct, learning_rate,
                       max_iterations, variables)
        return self.get_weights()

    def get_weights(self):
        if self.weight_cache is None:
            self.weight_cache = self.root.get_weights()
        return self.weight_cache

    def set_weights(self, new_weights):
        self.root.set_weights(new_weights)

    def derive(self, var_index, by_weights=True):
        derivative_id = 'X'
        if by_weights:
            derivative_id = 'W'
        derivative_id += f'_{var_index}'

        if self.derivative_cache.get(derivative_id) is None:
            derivative_root = self.root.derive(var_index, by_weights)
            derivative = Cell(derivative_root, self.arity, 0)
            self.derivative_cache[derivative_id] = derivative
            return derivative
        return self.derivative_cache[derivative_id]

    def __invert__(self):
        return ~self.root

    def clone(self) -> "Cell":
        return Cell(self.root.clone(), self.arity, self.depth)

    def __repr__(self):
        return (f"root = {self.to_python()}, age = {self.age}, marked? "
                f"= {self.marked}, fitness = {self.fitness}")

    def __str__(self):
        return (f"Individual: {self.to_python()} \n"
                f"Fitness: {self.get_fit()} \n"
                f"Age: {self.age} \n"
                f"Marked? {self.marked}\n"
                f"")

    def get_state_weight(self):
        if self.state_weight is None:
            self.state_weight = State(self)
        return self.state_weight


t_cell = TypeVar('t_cell', bound=Cell)
t_cell_list = TypeVar('t_cell_list', bound=List[Cell])
