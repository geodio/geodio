import numpy as np

from core.cell import Weight, Function, LinearTransformation, b_var, \
    SigmoidActivation


def make_weight(*args):
    return Weight(np.array(args))


dummy = [b_var()]


def handle_reserved(func_name, args):
    if func_name == "weight":
        f = Function(len(args), make_weight, args)
        return f
    if func_name == "Linear":
        return LinearTransformation(int(args[0]), int(args[0]), dummy)
    if func_name == "Sigmoid":
        return SigmoidActivation(dummy)
