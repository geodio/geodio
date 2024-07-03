import sys

import numpy as np

from src.cell.collections.builtin_functors import DEFAULT_FUNCTORS
from src.cell.optim.fitness import MSE
from src.genetic.generator import RandomOrganismGenerator, RandomCellGenerator
from src.genetic.pop_utils import PopulationProperties


def main():
    x = np.array([[1], [2], [3], [4], [5], [10]])
    y = np.array([
        [1 + 3.01],
        [2 ** 6.9 + 3.01],
        [3 ** 6.9 + 3],
        [4 ** 6.9 + 3.01],
        [5 ** 6.9 + 3.01],
        [10 ** 6.9 + 3.01]
    ])

    func_set = DEFAULT_FUNCTORS
    term_set = [0, 1, 2, 3, -1, 4, 6, 7]
    var_count = 1  # Number of input variables
    max_depth = 2  # Maximum depth of the tree
    output_arity = 7
    prop = PopulationProperties(10, func_set, term_set,
                                max_depth, var_count)

    cell_generator = RandomCellGenerator()
    cell_generator.set_pop_prop(prop)

    rog = RandomOrganismGenerator(3, 1, cell_generator,
                                  2,
                                  2,
                                  2,
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
    print(org.fitness)


if __name__ == '__main__':
    main()
