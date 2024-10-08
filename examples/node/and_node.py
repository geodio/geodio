import sys

import numpy as np

from python.geodio.core.cell.train.loss import MSEMultivariate
from python.geodio.core.cell.train.optimization_args import OptimizationArgs
from python.geodio.core.cell.train.optimizer import Optimizer
from python.geodio.core import SigmoidActivation
from python.geodio.core import Node


def main():
    dim_in = 2
    dim_out = 2
    arity = 1

    activation_function = SigmoidActivation()

    node1 = Node(arity, dim_in, dim_out, activation_function,
                 optimizer=Optimizer())
    node1.set_optimization_risk(True)
    input_data = [
        [np.array([1, 1])],
        [np.array([1, 0])],
        [np.array([0, 1])],
        [np.array([0, 0])],
    ]
    desired_output = [
        [np.array([1.0, 1.0])],
        [np.array([0.0, 1.0])],
        [np.array([0.0, 1.0])],
        [np.array([0.0, 0.0])],
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
    output_before = [node1(input_data_i) for input_data_i in input_data]
    print("Weights before optimization:")
    print([[w, w.w_index] for w in node1.get_weights()])
    print(loss_function.evaluate(node1, input_data, desired_output))
    node1.optimize(optimization_args)
    print("Weights after optimization:")
    print([[w, w.w_index] for w in node1.get_weights()])
    print(loss_function.evaluate(node1, input_data, desired_output))
    output = [node1(input_data_i) for input_data_i in input_data]

    print("            Input         :", input_data)
    print("Before Optimization output:", output_before)
    print("       Final output       :", output)
    print("       Desired output     :", desired_output)


if __name__ == "__main__":
    main()
