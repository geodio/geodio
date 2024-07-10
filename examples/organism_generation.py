import sys

import numpy as np

from core.cell.collections.neurons import NEURONS
from core.cell.optim.fitness import MSE
from core.genetic.generator import RandomOrganismGenerator, CellBankGenerator
from core.genetic.pop_utils import PopulationProperties


def main():
    x = np.array([
        [0, 10],
        [0, 0],
        [10, 0],
        # [10, 10]
    ])
    y = np.array([
        [0],
        [0],
        [1],
        # [0],
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

    rog = RandomOrganismGenerator(1, 2, cell_generator,
                                  (2, 10),
                                  (0, 0),
                                  (0, 0),
                                  10,
                                  output_arity)
    org = rog.generate_organism()
    org.optimize_values(
        MSE(),
        x,
        y,
        max_iterations=100,
        min_fitness=sys.maxsize
    )
    for xx in x:
        print(f"for input = {xx}, output =", org(xx))

    print(org.fitness)
    print(org)


if __name__ == '__main__':
    main()
