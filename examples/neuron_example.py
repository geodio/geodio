import sys

import numpy as np

from src.cell.fitness import MSE, get_predicted
from src.cell.neurons import Sigmoid


def main():
    X = np.array([[1], [2], [3], [4], [5]])
    Y = np.array([0, 0, 0, 1, 1])
    fitness = MSE()
    neuron = Sigmoid([1])

    neuron.optimize_values(fitness, X, Y, max_iterations=100, min_fitness=sys.maxsize)
    print("MSE:", fitness.evaluate(neuron, X, Y))
    print(get_predicted(X, neuron))
    print(neuron)



if __name__ == '__main__':
    main()
