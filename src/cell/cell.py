import pickle
import random
import sys
from typing import Optional

import numpy as np

from src import rnd
from src.cell.functors import Functors
from src.cell.operands.function import Function
from src.cell.operands.operand import Operand
import tensorflow as tf

from src.cell.operands.variable import Variable
from src.cell.operands.weight import Weight
from src.genetic.pop_utils import ReproductionPolicy


class Cell(Operand):
    def __init__(self, root: Operand, arity: int, max_depth,
                 reproduction_policy=ReproductionPolicy.DIVISION):
        super().__init__(arity)
        self.root = root
        self.depth = max_depth
        self.age = 0
        self.fitness = None
        self.marked = False
        self.reproduction_policy = reproduction_policy
        self.mutation_risk = 0.5

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

    def optimize_values(self, loss_function, optimizer, variables,
                        desired_output):
        with tf.GradientTape() as tape:
            weights = self.get_weights()
            tape.watch(weights)
            output = self(variables)
            loss_value = loss_function(output, desired_output)
        print(weights)
        gradient = tape.gradient(loss_value, weights)
        optimizer.apply_gradients(zip(gradient, weights))
        self.set_weights(weights)

    def get_weights(self):
        return self.root.get_weights()

    def set_weights(self, new_weights):
        self.root.set_weights(new_weights)

    def calculate_gradient(self, loss, variables, desired_output):
        with tf.GradientTape() as tape:
            # Forward pass: calculate the output of the tree
            output = self(variables)
            # Calculate the loss
            loss_value = loss(output, desired_output)
        # Use the tape to compute the gradient of the loss with respect to the
        # variables (values of "value" nodes)
        gradient = tape.gradient(loss_value, variables)

        return gradient

    def d(self, var_index) -> "Optional[Operand]":
        return self.root.d(var_index)

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
