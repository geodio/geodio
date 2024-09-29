import sys

import numpy as np


def fitness_mse(y, y_pred):
    # Mean Squared Error (MSE) fitness function
    x = np.mean((y - y_pred) ** 2)
    if str(x) == "nan" or str(x) == 'inf':
        x = sys.maxsize
    return x
