import sys

import numpy as np

from src.cell.fitness import MSE, get_predicted
from src.cell.neurons import Sigmoid


def main():
    X = [
        [np.array([1, 2])],
        [np.array([2, 3])],
        [np.array([3, 4])],
        [np.array([4, 5])],
        [np.array([5, 6])]
    ]
    Y = [0, 0, 0, 1, 1]
    fitness = MSE()
    neuron = Sigmoid(np.array([1, 2]))

    neuron.optimize_values(fitness, X, Y, max_iterations=10000,
                           min_fitness=sys.maxsize)
    print("MSE:", fitness.evaluate(neuron, X, Y))
    print(get_predicted(X, neuron))
    print(neuron)



if __name__ == '__main__':
    main()
