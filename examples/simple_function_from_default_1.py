import numpy as np

from src.cell.default_functors import DEFAULT_FUNCTORS
from src.cell.fitness import MSE
from src.pop import Pop


def main():
    # Example usage:
    x = np.array([[1], [2], [3], [4], [5], [10]])
    y = np.array([1 + 3.01, 2**6.9 + 3.01, 3**6.9 + 3, 4**6.9 + 3.01, 5**6.9
                  + 3.01,
                  10**6.9 +
                  3.01])


    func_set = DEFAULT_FUNCTORS
    term_set = [0, 1, 2, 3, -1, 4, 6, 7]
    var_count = 1  # Number of input variables
    max_depth = 2  # Maximum depth of the tree

    # Generate a random tree
    pop_size = 800
    generations = 1000

    pop = Pop(pop_size, func_set, term_set, max_depth, var_count,
              optimize=True)
    fitness_mse = MSE()
    best_ind, best_fit = pop.evolve_population(generations, x, y, fitness_mse)
    print(f"Best individual: {best_ind.to_python()}")
    print(f"Best fitness: {best_fit:.4f}")

def main2():
    # Example usage:
    x = np.array([[1], [2], [3], [4], [5], [10]])
    y = np.array([1 * 4.69, 2 * 4.69, 3 * 4.69, 4 * 4.69, 5 * 4.69, 10 * 4.69])


    func_set = DEFAULT_FUNCTORS
    term_set = [0, 1, 2, 3, -1, 4, 6, 6.9]
    var_count = 1  # Number of input variables
    max_depth = 2  # Maximum depth of the tree

    # Generate a random tree
    pop_size = 800
    generations = 1000

    pop = Pop(pop_size, func_set, term_set, max_depth, var_count,
              optimize=True)
    fitness_mse = MSE()
    best_ind, best_fit = pop.evolve_population(generations, x, y, fitness_mse)
    print(f"Best individual: {best_ind.to_python()}")
    print(f"Best fitness: {best_fit:.4f}")


if __name__ == '__main__':
    main()
