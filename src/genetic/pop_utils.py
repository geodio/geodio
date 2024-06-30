from enum import IntEnum


class PopulationProperties:
    def __init__(self, population_size, func_set, term_set, max_depth,
                 arity):
        self.population_size = population_size
        self.func_set = func_set
        self.term_set = term_set
        self.max_depth = max_depth
        self.arity = arity


class ReproductionPolicy(IntEnum):
    DIVISION = 0
    CROSSOVER = 1
