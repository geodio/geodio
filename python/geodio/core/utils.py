import numpy as np


def flatten(lst):
    """
    Flattens a list of any dimension to a 1-dimensional list.

    Parameters:
    - lst: The list to flatten.

    Returns:
    - A flattened 1-dimensional list.
    """
    flattened_list = []

    def _flatten(sublist):
        for item in sublist:
            if isinstance(item, (list, np.ndarray)):
                _flatten(item)
            else:
                flattened_list.append(item)

    _flatten(lst)
    return flattened_list
