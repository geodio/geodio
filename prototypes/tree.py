# import sys
#
# import tensorflow as tf
# import numpy as np
# import random
# import core.rnd as rnd
#
#
# class Node:
#     def __init__(self, node_type, value=None, children=None):
#         self.node_type = node_type  # "literal", "variable", "function"
#         self.value = value  # Value of literal or variable
#         self.children = children if children is not None else []
#
#     def add_child(self, child):
#         self.children.append(child)
#
#     def replace_child(self, old_child, new_child):
#         i = self.children.index(old_child)
#         self.children[i] = new_child
#
#     def run(self, variables):
#         if self.node_type == "literal":
#             return self.value
#         elif self.node_type == "variable":
#             return variables[self.value]
#         elif self.node_type == "function":
#             args = [child.run(variables) for child in self.children]
#             return self.value(*args)
#
#     def to_python(self):
#         if self.node_type == "literal":
#             return str(self.value)
#         elif self.node_type == "variable":
#             return f"x[{self.value}]"
#         elif self.node_type == "function":
#             args = [child.to_python() for child in self.children]
#             return f"{self.value.__name__}({', '.join(args)})"
#
#     def clone(self):
#         return Node(self.node_type, self.value, [kid.clone() for kid in self.children])
#
#
# class Tree:
#     def __init__(self, root: Node, arity: int, max_depth):
#         self.root = root
#         self.arity = arity
#         self.depth = max_depth
#         self.age = 0
#         self.fitness = None
#         self.marked = False
#
#     def nodes(self):
#         return self.root.children
#
#     def run(self, variables):
#         return self.root.run(variables)
#
#     def replace(self, node_old, node_new):
#         self.root.replace_child(node_old, node_new)
#
#     def to_python(self):
#         return self.root.to_python()
#
#     def mutate(self, func_set, term_set, max_depth=None, mutation_rate=0.1):
#         if not max_depth:
#             max_depth = self.depth
#         if np.random.rand() < mutation_rate and len(self.root.children) != 0:
#             mutant_node = create_random_node(max_depth - 1, term_set, func_set, self.arity)
#             self.depth = max_depth
#             self._randomly_replace(mutant_node)
#             self.fitness = None
#         return self
#
#     def _randomly_replace(self, mutant_node):
#         i = rnd.from_range(0, len(self.root.children), True)
#         self.root.children[i] = mutant_node
#
#     def get_age(self):
#         return self.age
#
#     def inc_age(self, age_benefit=0):
#         """
#         ages the tree if the fitness exists
#         :param age_benefit: contribution of age to the fitness
#         :return:
#         """
#         if self.fitness is not None:
#             self.age += 1
#             self.fitness *= (1 - age_benefit)
#
#     def mark(self):
#         self.marked = True
#
#     def get_fit(self):
#         return self.fitness if self.fitness is not None else sys.maxsize
#
#     def optimize_values(self, loss_function, optimizer, variables,
#                         desired_output):
#         # Calculate the output of the tree
#         output = self.run(variables)
#         # Calculate the loss
#         loss = loss_function(output, desired_output)
#         # Calculate the gradient of the loss with respect to the values of
#         # "value" nodes
#         gradient = self.calculate_gradient(loss, variables, desired_output)
#         # Update the values of "value" nodes using the optimizer
#         optimizer.update_values(gradient)
#
#     def calculate_gradient(self, loss, variables, desired_output):
#         with tf.GradientTape() as tape:
#             # Forward pass: calculate the output of the tree
#             output = self.run(variables)
#             # Calculate the loss
#             loss_value = loss(output, desired_output)
#         # Use the tape to compute the gradient of the loss with respect to the
#         # variables (values of "value" nodes)
#         gradient = tape.gradient(loss_value, variables)
#
#         return gradient
#
#     def __repr__(self):
#         return (f"root = {self.to_python()}, age = {self.age}, marked? "
#                 f"= {self.marked}, fitness = {self.fitness}")
#
#     def __str__(self):
#         return (f"Individual: {self.to_python()} \n"
#                 f"Fitness: {self.get_fit()} \n"
#                 f"Age: {self.age} \n"
#                 f"Marked? {self.marked}\n"
#                 f"")
#
#
# def create_random_node(depth, term_set, func_set, var_count):
#     if depth == 0 or random.random() < 0.3:  # Terminal node
#         node_type = "literal" if random.random() < 0.5 else "variable"
#         if node_type == "literal":
#             value = random.choice(term_set)
#         else:
#             value = random.randint(0, var_count - 1)
#         return Node(node_type, value=value)
#     else:  # Function node
#         func = rnd.choice(func_set)
#         node = Node("function", value=func)
#         arity = len(func.__code__.co_varnames)  # Number of arguments of the function
#         for _ in range(arity):
#             child = create_random_node(depth - 1, term_set, func_set, var_count)
#             node.add_child(child)
#         return node
#
#
# def generate_random(func_set, term_set, max_depth, var_count) -> Tree:
#     root = create_random_node(max_depth, term_set, func_set, var_count)
#     return Tree(root, var_count, max_depth)
#
#
# def crossover(tree1: 'Tree', tree2: 'Tree'):
#     # Perform crossover (recombination) operation
#     root1 = tree1.root
#     root2 = tree2.root
#     if len(root1.children) != 0 and len(root2.children) != 0:
#         index1 = rnd.from_range(0, len(root1.children), True)
#         index2 = rnd.from_range(0, len(root2.children), True)
#         # Choose random nodes from parents
#         node1 = root1.children[index1]
#         node2 = root2.children[index2]
#
#         # Swap the chosen nodes
#         new_child1 = node2.clone()
#         new_child2 = node1.clone()
#
#         # Replace the nodes in the children
#         root1.children[index1] = new_child1
#         root2.children[index2] = new_child2
#
#     return Tree(root1, tree1.arity, tree1.depth), Tree(root2, tree2.arity, tree1.depth)
#
#
# # Example usage
# def add(x, y):
#     return x + y
#
#
# def subtract(x, y):
#     return x - y
#
#
# def multiply(x, y):
#     return x * y
#
#
# def divide(x, y):
#     if y == 0:
#         return 0  # Avoid division by zero
#     return x / y
#
#
# def main():
#     func_set = [add, subtract, multiply, divide]
#     term_set = [1, 2, 3, 69]
#     var_count = 2  # Number of input variables
#     max_depth = 3  # Maximum depth of the tree
#     # Generate a random tree
#     random_tree = generate_random(func_set, term_set, max_depth, var_count)
#     print("Random Tree:")
#     print(random_tree.to_python())
#     # Evaluate the random tree
#     variables = [3, 4]  # Example input values
#     result = random_tree.run(variables)
#     print("Evaluation Result:", result)
#
#
# if __name__ == '__main__':
#     main()
