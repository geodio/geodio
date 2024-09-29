import numpy as np

from geodio.core.cell import Weight, Function, LinearTransformation, b_var, \
    SigmoidActivation, BOO, OptimizationArgs, MSEMultivariate


def make_weight(*args):
    return Weight(np.array(args))


dummy = [b_var()]


def make_linear(dim_in, dim_out):
    return LinearTransformation(int(dim_in), int(dim_out), dummy)


def make_sigmoid(*args):
    return SigmoidActivation(dummy)


def train(nn: BOO, inputs, output):
    # print(isinstance(inputs, np.ndarray))

    nn.set_optimization_risk(True)
    # print("HERE", nn([np.array([10, 10])]))
    args = OptimizationArgs(
        learning_rate=0.01,
        max_iter=1000,
        loss_function=MSEMultivariate(),
        inputs=inputs,
        desired_output=output,
        backpropagation=True,
        risk=True,
        decay_rate=0,
        ewc_lambda=0.0001,
        l2_lambda=0,
        epochs=100,
        batch_size=1,
        grad_reg='adam'
    )
    nn.optimize(args)


def handle_reserved(func_name, args):
    if func_name == "weight":
        f = Function(len(args), make_weight, args)
        return f
    if func_name == "Linear":
        return Function(len(args), make_linear, args)
    if func_name == "Sigmoid":
        return Function(len(args), make_sigmoid, args)
    if func_name == 'print':
        return lambda w: Function(len(args), print, args)
    if func_name == 'train':
        return lambda w: Function(len(args), train, args)
