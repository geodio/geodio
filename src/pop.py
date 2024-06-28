import sys

import numpy as np

from src.cell.cell import generate_random


class Pop:
    def __init__(self, pop_size, func_set, term_set, max_depth, arity,
                 kill_rate=0.1, crossover_rate=0.8, mutation_rate=0.2,
                 age_benefit=1e-4):
        self.pop_size = pop_size
        self.func_set = func_set
        self.term_set = term_set
        self.max_depth = max_depth
        self.arity = arity
        self.kill_rate = kill_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
        # Tracks the number of generations individuals have survived
        self.survival_counts = np.zeros(pop_size, dtype=int)
        self.age_benefit = age_benefit

    def initialize_population(self):
        # Initialize population with random expressions
        pop = self._new_offspring(self.pop_size)
        return pop

    def _new_offspring(self, size):
        return [generate_random(self.func_set, self.term_set, self.max_depth,
                                self.arity) for _ in range(size)]

    def evaluate_population(self, x, y, fitness_func):
        # Evaluate fitness of each individual in the population
        for cell in self.population:
            if cell.fitness is None:
                try:
                    y_pred = []
                    for x_inst in x:
                        y_pred.append(cell(x_inst))
                    if None in y_pred:
                        # Invalid cell
                        fit = sys.maxsize
                    else:
                        fit = np.abs(fitness_func(y, y_pred))
                except SyntaxError:
                    # Tree with syntax error
                    fit = sys.maxsize
                cell.fitness = fit
        sorted(self.population, key=lambda indi: indi.get_fit(), reverse=True)

    def age_all(self):
        for individual in self.population:
            individual.inc_age(self.age_benefit)

    def mark_best(self, threshold=1e-2):
        to_be_marked = [a for a in self.population if a.fitness < threshold]
        for x in to_be_marked:
            x.mark()

    def get_best_ind(self):
        # Find the best individual in the population
        best_idx = self._min_fit_idx()
        best = self.population[best_idx]
        return best, best.get_fit()

    def _min_fit_idx(self):
        return min(range(len(self.population)), key=lambda i: self.population[i].get_fit())

    def _max_fit_idx(self):
        return max(range(len(self.population)), key=lambda i: self.population[i].fitness)

    def _fit_of_idx(self, idx):
        return self.population[idx].fitness

    def kill_worst(self):
        fitness = np.array(list(map(lambda xx: xx.fitness, self.population)),
                           dtype=np.float_)
        max_fit = fitness[np.argmax(fitness)]
        min_fit = fitness[np.argmin(fitness)]
        t = (max_fit - min_fit) * self.kill_rate + min_fit
        to_be_killed = fitness > t
        population_array = np.array(self.population)
        population_array[to_be_killed] = self._new_offspring(sum(to_be_killed))
        self.population = population_array.tolist()

    def mutate_middle(self):
        """
        mutates if individual is not marked and if individual is not new offspring
        :return:
        """
        cond = lambda tree: not tree.marked and tree.fitness is not None
        middle_class = [a for a in self.population if cond(a)]
        for individual in middle_class:
            individual.mutate(self.func_set,
                              self.term_set,
                              self.max_depth,
                              self.mutation_rate)

    def grow_func(self, generations, x, y, fitness_func):
        """
        Genetic Expression Programming (GEP)

        Parameters:
        - generations: int
            Number of generations
        - x: array-like, shape (n_samples, 1)
            Independent variable
        - y: array-like, shape (n_samples, 1)
            Dependent variable
        - fitness_func: callable
            Fitness function (e.g., mean squared error)

        Returns:
        - best_ind: Tree
            Best individual (expression) found
        - best_fit: float
            Best fitness value found
        """
        self.evaluate_population(x, y, fitness_func)
        for gen in range(generations):
            self.mark_best()
            self.kill_worst()
            self.age_all()
            self.mutate_middle()
            self.evaluate_population(x, y, fitness_func)
            if gen % 20 == 0:
                print(f"====GENERATION {gen}====")
                best_ind, best_fit = self.get_best_ind()
                print(best_ind)

        best_ind, best_fit = self.get_best_ind()
        return best_ind, best_fit
