import pickle
import random
import sys

import numpy as np

from src import rnd
from src.cell.fitness import FitnessFunction
from src.cell.functors import Functors
from src.cell.operands.function import Function
from src.cell.operands.operand import Operand

from src.cell.operands.variable import Variable
from src.cell.operands.weight import Weight
from src.genetic.pop_utils import ReproductionPolicy
from src.optimizer import Optimizer


class Cell(Operand):
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
        self.derivative_cell = None
        if optimizer is None:
            self.optimizer = Optimizer()
        else:
            self.optimizer = optimizer

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

    def get_fit(self):
        return self.fitness if self.fitness is not None else sys.maxsize

    def optimize_values(self, fit_fct: FitnessFunction, variables,
                        desired_output,
                        learning_rate=0.1,
                        max_iterations=100,
                        min_fitness=10):
        y_pred = [self(x_inst) for x_inst in variables]
        self.fitness = fit_fct(desired_output, y_pred)
        if not (self.fitness <= min_fitness or self.marked):
            return

        max_iterations *= (1 / (self.age + 1))
        max_iterations = int(max_iterations)

        self.optimizer(self,
                       desired_output,
                       fit_fct,
                       learning_rate,
                       max_iterations,
                       variables)
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

    def to_bytes(self):
        return pickle.dumps(self)

    @staticmethod
    def from_bytes(data):
        return pickle.loads(data)

    def __repr__(self):
        return (f"root = {self.to_python()}, age = {self.age}, marked? "
                f"= {self.marked}, fitness = {self.fitness}")

    def __str__(self):
        return (f"Individual: {self.to_python()} \n"
                f"Fitness: {self.get_fit()} \n"
                f"Age: {self.age} \n"
                f"Marked? {self.marked}\n"
                f"")


def create_random_node(depth, term_set, func_set, var_count):
    if depth == 0 or random.random() < 0.3:  # Terminal node
        operand_type = "weight" if random.random() < 0.5 else "variable"
        if operand_type == "weight":
            value = random.choice(term_set)
            return Weight(value)
        value = random.randint(0, var_count - 1)
        return Variable(value)
    else:  # Function node
        node = generate_function_node(depth, func_set, term_set, var_count)
        return node


def generate_function_node(depth, func_set, term_set, var_count):
    if not isinstance(func_set, Functors):
        func = rnd.choice(func_set)
        # Number of arguments of the function
        arity = len(func.__code__.co_varnames)
        node = Function(arity=arity, value=func)
    else:
        node = func_set.get_random_clone()
        arity = node.arity
    for _ in range(arity):
        child = create_random_node(depth - 1, term_set, func_set, var_count)
        node.add_child(child)
    return node


def generate_random(func_set, term_set, max_depth, var_count) -> Cell:
    root = create_random_node(max_depth, term_set, func_set, var_count)
    return Cell(root, var_count, max_depth)


def crossover(left_cell: 'Cell', right_cell: 'Cell'):
    # Perform crossover (recombination) operation
    root1 = left_cell.root
    root2 = right_cell.root
    if len(root1.children) != 0 and len(root2.children) != 0:
        index1 = rnd.from_range(0, len(root1.children), True)
        index2 = rnd.from_range(0, len(root2.children), True)
        # Choose random nodes from parents
        node1 = root1.children[index1]
        node2 = root2.children[index2]

        # Swap the chosen nodes
        new_child1 = node2.clone()
        new_child2 = node1.clone()
        new_child2.age = 0
        new_child1.age = 0

        # Replace the nodes in the children
        root1.children[index1] = new_child1
        root2.children[index2] = new_child2

    return (Cell(root1, left_cell.arity, left_cell.depth),
            Cell(root2, right_cell.arity, left_cell.depth))
