import numpy as np

from python.geodio.core import fitness_mse
from python.geodio.core.genetic.pop import Pop


def main():
    # Example usage:
    x = np.array([[1], [2], [3], [4], [5], [100]])
    y = np.array([2, 4, 6, 8, 10, 200])

    # Example usage
    def add(_x, _y):
        return _x + _y

    def subtract(_x, _y):
        return _x - _y

    def multiply(_x, _y):
        return _x * _y

    def divide(_x, _y):
        if _y == 0:
            return 0  # Avoid division by zero
        return _x / _y

    func_set = [add, subtract, multiply, divide]
    term_set = [0, 1, 2, 3, -1]
    var_count = 1  # Number of input variables
    max_depth = 4  # Maximum depth of the tree

    # Generate a random tree
    pop_size = 50
    generations = 100

    pop = Pop(pop_size, func_set, term_set, max_depth, var_count)
    best_ind, best_fit = pop.evolve_population(generations, x, y, fitness_mse)
    print(f"Best individual: {best_ind.to_python()}")
    print(f"Best fitness: {best_fit:.4f}")


if __name__ == '__main__':
    main()
