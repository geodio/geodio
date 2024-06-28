import numpy as np

from src.cell.default_functors import DEFAULT_FUNCTORS
from src.main import fitness_mse
from src.pop import Pop


def main():
    # Example usage:
    x = np.array([[1], [2], [3], [4], [5], [10]])
    y = np.array([1, 2**7, 3**7, 4**7, 5**7, 10**7])


    func_set = DEFAULT_FUNCTORS
    term_set = [0, 1, 2, 3, -1, 4, 7]
    var_count = 1  # Number of input variables
    max_depth = 2  # Maximum depth of the tree

    # Generate a random tree
    pop_size = 50
    generations = 100

    pop = Pop(pop_size, func_set, term_set, max_depth, var_count)
    best_ind, best_fit = pop.grow_func(generations, x, y, fitness_mse)
    print(f"Best individual: {best_ind.to_python()}")
    print(f"Best fitness: {best_fit:.4f}")


if __name__ == '__main__':
    main()
