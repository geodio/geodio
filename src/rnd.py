import numpy.random as rand


def choice(arr):
    n = len(arr)
    return arr[from_range(0, n, True)]


def from_range(start=0, till=1, integer=False):
    if not integer:
        return rand.random() * (till - start) + start
    else:
        return int(rand.random() * (till - start) + start)
