import sys
import threading
import time

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.cell.cell import generate_random, Cell
from src.cell.fitness import FitnessFunction
from src.genetic.generator import RandomGenerator
from src.genetic.pop_utils import PopulationProperties, ReproductionPolicy

lock = threading.Lock()


def evolve_individual(cell: Cell, x, y, fitness_func, age_benefit,
                      optimize=True):
    if cell is None:
        print("ERROR")
    try:
        if optimize:
            cell.optimize_values(fitness_func, x, y)
    except SyntaxError:
        cell.fitness = sys.maxsize
    if cell.fitness is None:
        try:
            y_pred = [cell(x_inst) for x_inst in x]
            if None in y_pred:
                fit = sys.maxsize
            else:
                fit = np.abs(fitness_func(y, y_pred))
        except SyntaxError:
            fit = sys.maxsize
    else:
        fit = cell.fitness
    cell.inc_age(age_benefit)
    cell.fitness = fit
    return cell


class Pop:
    def __init__(self, pop_size, func_set, term_set, max_depth, arity,
                 kill_rate=0.3, crossover_rate=0.8, mutation_rate=0.2,
                 age_benefit=1e-8, generator=None, optimize=False):
        self.pop_prop = PopulationProperties(pop_size, func_set, term_set,
                                             max_depth, arity)
        self.generator = generator or RandomGenerator()
        self.generator.set_pop_prop(self.pop_prop)
        self.kill_rate = kill_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
        self.survival_counts = np.zeros(pop_size, dtype=int)
        self.age_benefit = age_benefit
        self.optimize = optimize

    def initialize_population(self):
        pop = self.generator.new_offspring_list(self.pop_prop.population_size)
        return pop

    def _new_offspring(self, size):
        return [generate_random(self.pop_prop.func_set, self.pop_prop.term_set,
                                self.pop_prop.max_depth,
                                self.pop_prop.arity) for _ in range(size)]

    def _crossover_offspring(self, size):
        return [generate_random(self.pop_prop.func_set, self.pop_prop.term_set,
                                self.pop_prop.max_depth,
                                self.pop_prop.arity) for _ in range(size)]

    def evaluate_population(self, x, y, fitness_func):
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(evolve_individual, cell, x, y,
                                       fitness_func,
                                       self.age_benefit,
                                       self.optimize): cell for cell in
                       self.population}
            for future in as_completed(futures):
                future.result()
        self.population.sort(key=lambda indi: indi.get_fit(), reverse=True)

    def mark_best(self):
        mark_ratio = self.kill_rate / 100
        to_mark = int(len(self.population) * mark_ratio)
        to_be_marked = self.get_best_individuals(to_mark)
        for x in to_be_marked:
            x.mark()
            x.reproduction_policy = ReproductionPolicy.CROSSOVER

    def get_best_individuals(self, size):
        sorted_population = sorted(self.population, key=lambda indi:
        indi.fitness)[:size]
        sorted_population.extend([ind for ind in self.population if
                                  ind.fitness < 1e-2])
        return sorted_population

    def get_best_ind(self):
        best_idx = self._min_fit_idx()
        best = self.population[best_idx]
        return best, best.get_fit()

    def _min_fit_idx(self):
        return min(range(len(self.population)),
                   key=lambda i: self.population[i].get_fit())

    def _max_fit_idx(self):
        return max(range(len(self.population)),
                   key=lambda i: self.population[i].fitness)

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
        population_array[to_be_killed] = self.generator.reproduce_cells(
            self.get_marked_individuals(),
            sum(to_be_killed)
        )
        self.population = population_array.tolist()
        self.population = list(filter(lambda cell: cell is not None,
                                      self.population))

    def mutate_middle(self):
        """
        mutates if individual is not marked and if individual is not new offspring
        :return:
        """
        cond = lambda tree: not tree.marked and tree.fitness is not None
        middle_class = [a for a in self.population if cond(a)]
        for individual in middle_class:
            individual.mutate(self.generator,
                              self.pop_prop.max_depth)
            individual.age = 0

    def evolve_population(self, generations, x, y,
                          fitness_func: FitnessFunction):
        self.evaluate_population(x, y, fitness_func)
        for gen in range(generations):
            self.mark_best()
            self.kill_worst()
            self.mutate_middle()
            self.evaluate_population(x, y, fitness_func)
            if gen % 20 == 0:
                print(f"====GENERATION {gen}====")
                best_ind, best_fit = self.get_best_ind()
                print(best_ind)
        best_ind, best_fit = self.get_best_ind()
        return best_ind, best_fit

    def get_marked_individuals(self):
        return list(filter(lambda ind: ind.marked, self.population))
