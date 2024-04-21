import numpy as np
from genetic import (
    initialize_population,
    evaluate_population,
    select_parents,
    crossover,
    mutate,
    find_best_individual,
)
from tree import Tree, generate_random


def fitness_mse(y, y_pred):
    # Mean Squared Error (MSE) fitness function
    return np.mean((y - y_pred) ** 2)


def gep(pop_size, generations, func_set, term_set, x, y, fitness_func, max_depth=5, arity=1):
    """
    Genetic Expression Programming (GEP)

    Parameters:
    - pop_size: int
        Population size
    - generations: int
        Number of generations
    - func_set: list of str
        Function set (e.g., ['+', '-', '*', '/'])
    - term_set: list of str
        Terminal set (e.g., ['x', 'y', '1', '2'])
    - x: array-like, shape (n_samples, 1)
        Independent variable
    - y: array-like, shape (n_samples, 1)
        Dependent variable
    - fitness_func: callable
        Fitness function (e.g., mean squared error)
    - max_depth: int, default=5
        Maximum tree depth

    Returns:
    - best_ind: str
        Best individual (expression) found
    - best_fit: float
        Best fitness value found
    """
    pop = initialize_population(pop_size, func_set, term_set, max_depth, arity)
    fitness = None
    for _ in range(generations):
        fitness = evaluate_population(pop, x, y, fitness_func)
        parents = select_parents(pop, fitness)
        offspring = crossover(parents, func_set, term_set, max_depth)
        pop = mutate(offspring, func_set, term_set, max_depth)
    best_ind, best_fit = find_best_individual(pop, fitness)
    return best_ind, best_fit


def main():
    # Example usage:
    x = np.array([1, 2, 3, 4, 5])
    y = np.array([2, 4, 6, 8, 10])

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

    func_set = [add, subtract, multiply, divide]
    term_set = [1, 2, 3, 69]
    var_count = 2  # Number of input variables
    max_depth = 3  # Maximum depth of the tree

    # Generate a random tree
    pop_size = 50
    generations = 100
    best_ind, best_fit = gep(pop_size, generations, func_set, term_set, x, y, fitness_mse, arity = 1)
    print(f"Best individual: {best_ind.to_python()}")
    print(f"Best fitness: {best_fit:.4f}")


if __name__ == '__main__':
    main()
