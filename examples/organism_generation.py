import sys

import numpy as np

from core.cell import NEURONS
from core.cell import CheckpointedMSE
from core.genetic.generator import CellBankGenerator
from core.genetic.pop_utils import PopulationProperties


def main():
    x = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
        [0, 0],
        [0, 1],
#        [1, 0]
    ])
    y = np.array([
        [[0]],
        [[1]],
        [[1]],
        [[0]],
        [[0]],
        [[1]],
#        [[1]]
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

    rog = RandomOrganismGenerator(2, 2, cell_generator,
                                  (2, 10),
                                  (0, 0),
                                  (0, 0),
                                  2,
                                  output_arity)
    org = rog.generate_organism()
    print(org.to_python())
    org.optimize_values(
        CheckpointedMSE(),
        x,
        y,
        max_iterations=100,
        min_error=sys.maxsize
    )
    for xx in x:
        print(f"for input = {xx}, output =", org(xx))

    print(org.error)
    print(org)

if __name__ == '__main__':
    main()
