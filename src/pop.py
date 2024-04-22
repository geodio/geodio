import numpy as np
from tree import Tree, generate_random, crossover as cross


class Pop:
    def __init__(self, pop_size, func_set, term_set, max_depth, arity, kill_rate = 0.3):
        self.pop_size = pop_size
        self.func_set = func_set
        self.term_set = term_set
        self.max_depth = max_depth
        self.arity = arity
        self.kill_rate = kill_rate
        self.population = self.initialize_population()

    def initialize_population(self):
        # Initialize population with random expressions
        pop = []
        for _ in range(self.pop_size):
            tree = generate_random(self.func_set, self.term_set, self.max_depth, self.arity)
            pop.append(tree)
        return pop

    def evaluate_population(self, x, y, fitness_func):
        # Evaluate fitness of each individual in the population
        fitness = []
        for tree in self.population:
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


    def get_best_ind(self, fitness):
        # Find the best individual in the population
        best_idx = np.argmin(fitness)
        return self.population[best_idx], fitness[best_idx]

    def kill_worst(self, fitness):
        max_fit = np.max(fitness)
        min_fit = np.min(fitness)
        t = (max_fit - min_fit) * self.kill_rate + min_fit
        to_be_killed = np.where(fitness > t)


        worst_idx = np.argsort(fitness)[-len(offspring):]
        pass
