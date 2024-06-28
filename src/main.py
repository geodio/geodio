import numpy as np

from genetic import (
    initialize_population,
    evaluate_population,
    select_parents,
    crossover,
    mutate,
    find_best_individual,
)
from src.pop import Pop


def fitness_mse(y, y_pred):
    # Mean Squared Error (MSE) fitness function
    return np.mean((y - y_pred) ** 2)


def gep(pop_size, generations, func_set, term_set, x, y, fitness_func,
        max_depth=5, arity=1):
    """
    Genetic Expression Programming (GEP)

    Parameters:
    - pop_size: int
        Population size
    - generations: int
        Number of generations
    - func_set: list of str
        Function set
    - term_set: list of str
        Terminal set (e.g., [True, 'k', 1, 2])
    - x: array-like, shape (n_samples, 1)
        Independent variable
    - y: array-like, shape (n_samples, 1)
        Dependent variable
    - fitness_func: callable
        Fitness function (e.g., mean squared error)
    - max_depth: int, default=5
        Maximum tree depth

    Returns:
    - best_ind: Tree
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

    pop = Pop(pop_size,func_set,term_set,max_depth,var_count)
    best_ind, best_fit = pop.grow_func(generations,x,y,fitness_mse)
    print(f"Best individual: {best_ind.to_python()}")
    print(f"Best fitness: {best_fit:.4f}")


def main2():
    # Example usage:
    x = np.array([[1], [4], [9], [16], [25], [100]])
    y = np.array([7, 14, 21, 28, 35, 70])

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
    max_depth = 4  # Maximum depth of the tree

    # Generate a random tree
    pop_size = 50
    generations = 300
    # best_ind, best_fit = gep(
    #     pop_size,
    #     generations,
    #     func_set,
    #     term_set,
    #     x, y,
    #     fitness_mse,
    #     max_depth,
    #     arity=1
    # )
    pop = Pop(pop_size,func_set,term_set,max_depth,var_count)
    best_ind, best_fit = pop.grow_func(generations,x,y,fitness_mse)
    print(f"Best individual: {best_ind.to_python()}")
    print(f"Best fitness: {best_fit:.4f}")


if __name__ == '__main__':
    main2()
