import numpy as np

from src.pop import Pop
from tree import Tree, generate_random, crossover as cross


def initialize_population(pop_size, func_set, term_set, max_depth, arity):
    # Initialize population with random expressions
    pop = []
    for _ in range(pop_size):
        tree = generate_random(func_set, term_set, max_depth, arity)
        pop.append(tree)
    return pop


def evaluate_population(pop, x, y, fitness_func):
    # Evaluate fitness of each individual in the population
    fitness = []
    for tree in pop:
        try:
            y_pred = tree.evaluate(x)
            if y_pred is None:
                fit = 0  # Return fitness of 0 for invalid trees
            else:
                fit = fitness_func(y, y_pred)
        except SyntaxError:
            fit = 0  # Return fitness of 0 for trees with syntax errors
        fitness.append(fit)
    return fitness


def replace_worst(pop, fitness, offspring, offspring_fitness):
    # Find indices of worst individuals
    worst_idx = np.argsort(fitness)[-len(offspring):]

    # Replace worst individuals with offspring
    for i, idx in enumerate(worst_idx):
        pop[idx] = offspring[i]

    return pop


def select_parents(pop, fitness, tournament_size=4):
    # Select parents using tournament selection
    # Shuffle the population indexes

    indexes = np.random.permutation(len(pop))

    parents = []
    n = len(pop) - tournament_size
    if n > 0:
        for i in range(n):
            tournament = indexes[i: i + tournament_size]  # Select tournament members from shuffled indexes
            winner = np.argmin([fitness[i] for i in tournament])
            parents.append(pop[tournament[winner]])
        return parents
    else:
        return pop


def crossover(parents, func_set, term_set, max_depth):
    # Perform crossover (recombination) operation
    offspring = []
    for _ in range(len(parents) // 2):
        parent1, parent2 = np.random.choice(parents, 2, replace=False)
        child1, child2 = crossover_tree(parent1, parent2, func_set, term_set, max_depth)
        offspring.extend([child1, child2])
    return offspring


def crossover_tree(parent1, parent2, func_set, term_set, max_depth):
    # Perform crossover (recombination) operation on two trees
    if np.random.rand() < 0.5:  # 50% chance of swapping subtrees
        cross(parent1, parent2)
    return parent1, parent2


def mutate(offspring, func_set, term_set, max_depth, mutation_rate=0.1):
    # Perform mutation operation
    for i in range(len(offspring)):
        if np.random.rand() < mutation_rate:
            offspring[i] = mutate_tree(offspring[i], func_set, term_set, max_depth)
    return offspring


def mutate_tree(tree, func_set, term_set, max_depth):
    # Perform mutation operation on a tree

    return tree.mutate(func_set, term_set, max_depth)


def find_best_individual(pop, fitness):
    # Find the best individual in the population
    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx]


def grow_func(pop_size, generations, func_set, term_set, x, y, fitness_func, max_depth=5, arity=1):
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

    pop = Pop(pop_size, func_set, term_set, max_depth, arity)
    fitness = None
    for _ in range(generations):
        fitness = pop.evaluate_population(x, y, fitness_func)
        parents = select_parents(pop, fitness)
        offspring = crossover(parents, func_set, term_set, max_depth)
        pop = mutate(offspring, func_set, term_set, max_depth)
    best_ind, best_fit = pop.get_best_ind(fitness)
    return best_ind, best_fit
