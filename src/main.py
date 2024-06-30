import numpy as np

from src.genetic_util import (
    initialize_population,
    evaluate_population,
    select_parents,
    crossover,
    mutate,
    find_best_individual,
)


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


