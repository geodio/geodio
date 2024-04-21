import numpy as np
import random
import rnd

class Node:
    def __init__(self, node_type, value=None, children=None):
        self.node_type = node_type  # "literal", "variable", "function"
        self.value = value  # Value of literal or variable
        self.children = children if children is not None else []

    def add_child(self, child):
        self.children.append(child)

    def replace_child(self, old_child, new_child):
        i = self.children.index(old_child)
        self.children[i] = new_child

    def evaluate(self, variables):
        if self.node_type == "literal":
            return self.value
        elif self.node_type == "variable":
            return variables[self.value]
        elif self.node_type == "function":
            args = [child.evaluate(variables) for child in self.children]
            return self.value(*args)

    def to_python(self):
        if self.node_type == "literal":
            return str(self.value)
        elif self.node_type == "variable":
            return f"x[{self.value}]"
        elif self.node_type == "function":
            args = [child.to_python() for child in self.children]
            return f"{self.value.__name__}({', '.join(args)})"

    def clone(self):
        return Node(self.node_type, self.value, [kid.clone() for kid in self.children])


class Tree:
    def __init__(self, root: Node, arity: int, max_depth):
        self.root = root
        self.arity = arity
        self.depth = max_depth

    def nodes(self):
        return self.root.children

    def evaluate(self, variables):
        return self.root.evaluate(variables)

    def replace(self, node_old, node_new):
        self.root.replace_child(node_old, node_new)

    def to_python(self):
        return self.root.to_python()

    def mutate(self, func_set, term_set, max_depth=None):
        if not max_depth:
            max_depth = self.depth
        if np.random.rand() < 0.5 and len(self.root.children) != 0:  # 50% chance of replacing a subtree

            mutant_node = create_random_node(max_depth - 1, term_set, func_set,  self.arity)
            self.depth = max_depth
            self.randomly_replace(mutant_node)
        return self

    def randomly_replace(self, mutant_node):
        i = rnd.from_range(0, len(self.root.children), True)
        self.root.children[i] = mutant_node


def create_random_node(depth, term_set, func_set, var_count):
    if depth == 0 or random.random() < 0.3:  # Terminal node
        node_type = "literal" if random.random() < 0.5 else "variable"
        if node_type == "literal":
            value = random.choice(term_set)
        else:
            value = random.randint(0, var_count - 1)
        return Node(node_type, value=value)
    else:  # Function node
        func = rnd.choice(func_set)
        node = Node("function", value=func)
        arity = len(func.__code__.co_varnames)  # Number of arguments of the function
        for _ in range(arity):
            child = create_random_node(depth - 1, term_set, func_set, var_count)
            node.add_child(child)
        return node


def generate_random(func_set, term_set, max_depth, var_count):
    root = create_random_node(max_depth, term_set, func_set, var_count)
    return Tree(root, var_count, max_depth)


def crossover(tree1: 'Tree', tree2: 'Tree'):
    # Perform crossover (recombination) operation
    root1 = tree1.root
    root2 = tree2.root
    if len(root1.children) != 0 and len(root2.children) != 0:
        index1 = rnd.from_range(0, len(root1.children), True)
        index2 = rnd.from_range(0, len(root2.children), True)
        # Choose random nodes from parents
        node1 = root1.children[index1]
        node2 = root2.children[index2]

        # Swap the chosen nodes
        new_child1 = node2.clone()
        new_child2 = node1.clone()

        # Replace the nodes in the children
        root1.children[index1] = new_child1
        root2.children[index2] = new_child2

    return Tree(root1, tree1.arity, tree1.depth), Tree(root2, tree2.arity, tree1.depth)


# Example usage
def add(x, y):
    return x + y


def subtract(x, y):
    return x - y


def multiply(x, y):
    return x * y


def divide(x, y):
    if y == 0:
        return 0  # Avoid division by zero
    return x / y


def main():
    func_set = [add, subtract, multiply, divide]
    term_set = [1, 2, 3, 69]
    var_count = 2  # Number of input variables
    max_depth = 3  # Maximum depth of the tree
    # Generate a random tree
    random_tree = generate_random(func_set, term_set, max_depth, var_count)
    print("Random Tree:")
    print(random_tree.to_python())
    # Evaluate the random tree
    variables = [3, 4]  # Example input values
    result = random_tree.evaluate(variables)
    print("Evaluation Result:", result)


if __name__ == '__main__':
    main()
