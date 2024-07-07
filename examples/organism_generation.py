import sys

import numpy as np

from src.cell.collections.builtin_functors import DEFAULT_FUNCTORS
from src.cell.collections.neurons import NEURONS
from src.cell.optim.fitness import MSE
from src.genetic.generator import RandomOrganismGenerator, RandomCellGenerator, \
    CellBankGenerator
from src.genetic.pop_utils import PopulationProperties


def main():
    x = np.array([
        [0, 10],
        [10, 0],
        [0, 0],
        [10, 10]
    ])
    y = np.array([
        [1],
        [1],
        [0],
        [0],
    ])

    func_set = NEURONS
    term_set = [0, 1, 2, 3, -1, 4, 6, 7]
    var_count = 1  # Number of input variables
    max_depth = 2  # Maximum depth of the tree
    output_arity = 1
    prop = PopulationProperties(10, func_set, term_set,
                                max_depth, var_count)

    cell_generator = CellBankGenerator(func_set)
    cell_generator.set_pop_prop(prop)

    rog = RandomOrganismGenerator(3, 2, cell_generator,
                                  (2, 4),
                                  (0, 0),
                                  (0, 0),
                                  30,
                                  output_arity)
    org = rog.generate_organism()
    org.optimize_values(
        MSE(),
        x,
        y,
        max_iterations=100,
        min_fitness=sys.maxsize
    )
    for xx in reversed(x):
        print(f"for input = {xx}, output =", org(xx))

    print(org.fitness)


if __name__ == '__main__':
    main()
