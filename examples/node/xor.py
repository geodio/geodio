import sys

import numpy as np

from core.cell.collections.builtin_functors import Linker
from core.cell.optim.loss import MSEMultivariate
from core.cell.optim.optimization_args import OptimizationArgs
from core.cell.optim.optimizer import Optimizer
from core.organism.activation_function import SigmoidActivation
from core.organism.node import Node


def main():
    # TODO chaining not working
    dim_in = 2
    dim_mid = 2
    dim_out = 2
    arity = 1

    activation_function = SigmoidActivation()

    node2 = Node(arity, dim_mid, dim_out, activation_function,
                 optimizer=Optimizer())
    node1 = Node(arity, dim_in, dim_mid, activation_function,
                 optimizer=Optimizer())
    node1.set_optimization_risk(True)
    node2.set_optimization_risk(True)
    leenk = Linker(1, node2, node1, dim_in, True)
    leenk.set_optimization_risk(True)
    input_data = [
        [np.array([1, 1])],
        [np.array([0, 1])],
        [np.array([1, 0])],
        [np.array([0, 0])]
    ]
    desired_output = [
        [np.array([0.0, 0.0])],
        [np.array([1.0, 0.0])],
        [np.array([1.0, 0.0])],
        [np.array([0.0, 0.0])]
    ]
    loss_function = MSEMultivariate()

    optimization_args = OptimizationArgs(
        inputs=input_data,
        desired_output=desired_output,
        loss_function=loss_function,
        learning_rate=0.1,
        max_iter=10000,
        min_error=sys.maxsize
    )
    output_before = [leenk(input_data_i) for input_data_i in input_data]
    print("Weights before optimization:")
    print([[w, w.w_index] for w in leenk.get_weights()])
    print(loss_function.evaluate(leenk, input_data, desired_output))
    leenk.optimize(optimization_args)
    print("Weights after optimization:")
    print([[w, w.w_index] for w in leenk.get_weights()])
    print(loss_function.evaluate(leenk, input_data, desired_output))
    output = [leenk(input_data_i) for input_data_i in input_data]

    print("            Input         :", input_data)
    print("Before Optimization output:", output_before)
    print("       Final output       :", output)
    print("       Desired output     :", desired_output)


if __name__ == "__main__":
    main()
