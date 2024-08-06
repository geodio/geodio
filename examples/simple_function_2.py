import numpy as np

from core.cell import MSE
from core.genetic.pop import Pop


def main2():
    # Example usage:
    x = np.array([[1], [4], [9], [16], [25], [100], [10000]])
    y = np.array([7, 14, 21, 28, 35, 70, 700])

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

    def sqrt(x):
        return np.sqrt(np.abs(x))

    func_set = [add, subtract, multiply, divide, sqrt]
    term_set = [0, 1, 2, 3, -1, 5, 7]
    var_count = 1  # Number of input variables
    max_depth = 3  # Maximum depth of the tree

    # Generate a random tree
    pop_size = 500
    generations = 300
    fitness_mse = MSE()
    pop = Pop(pop_size, func_set, term_set, max_depth, var_count)
    best_ind, best_fit = pop.evolve_population(generations, x, y,
                                               fitness_mse)
    print(f"Best individual: {best_ind.to_python()}")
    print(f"Best fitness: {best_fit:.4f}")


if __name__ == '__main__':
    main2()
