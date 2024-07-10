from core.cell.cell import Cell
from core.math import rnd


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
