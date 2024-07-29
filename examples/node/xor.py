import sys

import numpy as np

from core.cell.collections.builtin_functors import Linker
from core.cell.optim.loss import MSEMultivariate
from core.cell.optim.optimization_args import OptimizationArgs
from core.cell.optim.optimizer import Optimizer
from core.organism.activation_function import SigmoidActivation
from core.organism.node import Node


def main():
    dim_in = 2
    dim_mid = 10
    dim_out = 1
    arity = 1

    activation_function = SigmoidActivation()

    node2 = make_node(activation_function, arity, dim_mid, dim_out)
    node1 = make_node(activation_function, arity, dim_in, dim_mid)
    leenk = link_nodes(dim_in, node1, node2)
    desired_output, input_data = xor_data()
    loss_function = MSEMultivariate()

    optimization_args = OptimizationArgs(
        inputs=input_data,
        desired_output=desired_output,
        loss_function=loss_function,
        learning_rate=0.1,
        max_iter=10000,
        min_error=sys.maxsize
    )
    train(desired_output, input_data, leenk, loss_function, optimization_args)


def main2():
    dim_in = 2
    dim_mid = 10
    dim_out = 1
    arity = 1

    activation_function = SigmoidActivation()

    node3 = make_node(activation_function, arity, dim_mid, dim_out)
    node2 = make_node(activation_function, arity, dim_mid, dim_mid)
    node1 = make_node(activation_function, arity, dim_in, dim_mid)
    leenk0 = link_nodes(dim_in, node1, node2)
    leenk = link_nodes(dim_in, leenk0, node3)
    desired_output, input_data = xor_data()
    loss_function = MSEMultivariate()

    optimization_args = OptimizationArgs(
        inputs=input_data,
        desired_output=desired_output,
        loss_function=loss_function,
        learning_rate=0.1,
        max_iter=10000,
        min_error=sys.maxsize
    )
    train(desired_output, input_data, leenk, loss_function, optimization_args)


def link_nodes(dim_in, node1, node2):
    leenk = Linker(1, node2, node1, dim_in, True)
    leenk.set_optimization_risk(True)
    return leenk


def make_node(activation_function, arity, dim_in, dim_out):
    node = Node(arity, dim_in, dim_out, activation_function,
                optimizer=Optimizer())
    node.set_optimization_risk(True)
    return node


def xor_data():
    input_data = [
        [np.array([1, 1])],
        [np.array([0, 1])],
        [np.array([1, 0])],
        [np.array([0, 0])]
    ]
    desired_output = [
        [np.array([0.0])],
        [np.array([1.0])],
        [np.array([1.0])],
        [np.array([0.0])]
    ]
    return desired_output, input_data


def train(desired_output, input_data, leenk, loss_function, optimization_args):
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
    main2()
