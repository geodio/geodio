import sys
import threading

import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed

from core import logger
from core.cell.cell import Cell
from core.cell.optim.loss import LossFunction
from core.genetic.generator import RandomCellGenerator
from core.genetic.pop_utils import PopulationProperties, ReproductionPolicy

lock = threading.Lock()


def evolve_individual(cell: Cell, x, y, fitness_func, age_benefit,
                      optimize=True, max_iterations=100, min_error=10):
    if cell is None or cell.root is None:
        logger.logging.error("Cannot evolve empty cell")
    try:
        if optimize:
            cell.optimize_values(fitness_func, x, y,
                                 max_iterations=max_iterations,
                                 min_error=min_error)
    except SyntaxError:
        cell.error = sys.maxsize
    if cell.error is None:
        try:
            y_pred = [cell(x_inst) for x_inst in x]
            if None in y_pred:
                fit = sys.maxsize
            else:
                fit = np.abs(fitness_func(y, y_pred))
        except SyntaxError:
            fit = sys.maxsize
    else:
        fit = cell.error
    cell.inc_age(age_benefit)
    cell.error = fit
    return cell


class Pop:
    def __init__(self, pop_size, func_set, term_set, max_depth, arity,
                 kill_rate=0.3, crossover_rate=0.8, mutation_rate=0.2,
                 age_benefit=1e-8, generator=None, optimize=False,
                 min_error=10):
        self.generations = 0
        self.pop_prop = PopulationProperties(pop_size, func_set, term_set,
                                             max_depth, arity)
        self.generator = generator or RandomCellGenerator()
        self.generator.set_pop_prop(self.pop_prop)
        self.kill_rate = kill_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.population = self.initialize_population()
        self.survival_counts = np.zeros(pop_size, dtype=int)
        self.age_benefit = age_benefit
        self.optimize = optimize
        self.mark_ratio = self.kill_rate / np.sqrt(
            self.pop_prop.population_size)
        self.min_error = min_error

    def initialize_population(self):
        pop = self.generator.new_offspring_list(self.pop_prop.population_size)
        return pop

    def evaluate_population(self, x, y, fitness_func):
        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    evolve_individual,
                    cell, x, y,
                    fitness_func,
                    self.age_benefit,
                    self.optimize, self.generations,
                    self.min_error
                ): cell for cell in self.population
            }
            for future in as_completed(futures):
                future.result()
        self.population.sort(key=lambda indi: indi.get_error(), reverse=True)

    def mark_best(self):
        to_mark = int(len(self.population) * self.mark_ratio)
        to_be_marked = self.get_best_individuals(to_mark)
        for x in to_be_marked:
            x.mark()
            x.reproduction_policy = ReproductionPolicy.CROSSOVER

    def get_best_individuals(self, size):
        sorted_population = sorted(
            self.population, key=lambda indi: indi.error)[:size]
        sorted_population.extend([ind for ind in self.population if
                                  ind.error < 1])
        return sorted_population

    def get_best_ind(self):
        best_idx = self._min_fit_idx()
        best = self.population[best_idx]
        return best, best.get_error()

    def _min_fit_idx(self):
        return min(range(len(self.population)),
                   key=lambda i: self.population[i].get_error())

    def _max_fit_idx(self):
        return max(range(len(self.population)),
                   key=lambda i: self.population[i].error)

    def _fit_of_idx(self, idx):
        return self.population[idx].error

    def kill_worst(self):
        fitness = np.array(list(map(lambda xx: xx.error, self.population)),
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
        cond = lambda tree: not tree.marked and tree.error is not None
        middle_class = [a for a in self.population if cond(a)]
        for individual in middle_class:
            individual.mutate(self.generator,
                              self.pop_prop.max_depth)
            individual.age = 0

    def evolve_population(self, generations, x, y,
                          fitness_func: LossFunction):
        self.evaluate_population(x, y, fitness_func)
        self.generations = generations
        for gen in range(generations):
            self.mark_best()
            self.kill_worst()
            self.mutate_middle()
            self.evaluate_population(x, y, fitness_func)
            if gen % 20 == 0:
                logger.logging.debug(f"====GENERATION {gen}====")
                best_ind, best_fit = self.get_best_ind()
                logger.logging.debug(best_ind)
                if best_fit < 1e-5:
                    break
        best_ind, best_fit = self.get_best_ind()
        return best_ind, best_fit

    def get_marked_individuals(self):
        return list(filter(lambda ind: ind.marked, self.population))
