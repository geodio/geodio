import sys

import numpy as np

from core.cell.collections.builtin_functors import Linker
from core.cell.optim.loss import MSEMultivariate
from core.cell.optim.optimization_args import OptimizationArgs
from core.cell.optim.optimizer import Optimizer
from core.organism.activation_function import SigmoidActivation
from core.organism.node import Node
from core.organism.organism import Organism


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


def xor_2_data():
    input_data = [
        [np.array([1, 1])],
        [np.array([0, 1])],
        [np.array([1, 0])],
        [np.array([0, 0])]
    ]
    desired_output = [
        [np.array([1.0])],
        [np.array([0.0])],
        [np.array([0.0])],
        [np.array([1.0])]
    ]
    return desired_output, input_data


def iris_sepal():
    input_data = [
        [np.array([5, 3.4, 0])],  # setosa
        [np.array([4.6, 3.1, 0])],  # setosa
        [np.array([5.5, 2.3, 0])],  # versicolor
        [np.array([6.5, 2.8, 0])],  # versicolor
        [np.array([7.7, 3.8, 0])],  # virginica
    ]
    desired_output = [
        [np.array([.5])],
        [np.array([.5])],
        [np.array([0])],
        [np.array([0])],
        [np.array([1])],
    ]
    return desired_output, input_data


def dummy_data():
    input_data = [
        [np.array([1, 1])],
    ]
    desired_output = [
        [np.array([0.5])],
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


def launch_model(model, max_iter=10000, data='xor'):
    if data == 'xor':
        desired_output, input_data = xor_data()
    elif data == 'xor_2':
        desired_output, input_data = xor_2_data()
    elif data == 'iris':
        desired_output, input_data = iris_sepal()
    else:
        desired_output, input_data = dummy_data()
    loss_function = MSEMultivariate()

    optimization_args = OptimizationArgs(
        inputs=input_data,
        desired_output=desired_output,
        loss_function=loss_function,
        learning_rate=0.1,
        max_iter=max_iter,
        min_error=sys.maxsize
    )
    train(desired_output, input_data, model, loss_function, optimization_args)


def link_nodes(dim_in, node1, node2):
    leenk = Linker(node2, node1, dim_in, True)
    leenk.set_optimization_risk(True)
    return leenk


def make_node(activation_function, arity, dim_in, dim_out):
    node = Node(arity, dim_in, dim_out, activation_function,
                optimizer=Optimizer())
    node.set_optimization_risk(True)
    return node


def get_link_model(activation_function, arity, dim_in, dim_mid, dim_out):
    node3 = make_node(activation_function, arity, dim_mid, dim_out)
    node2 = make_node(activation_function, arity, dim_mid, dim_mid)
    node1 = make_node(activation_function, arity, dim_in, dim_mid)
    model = link_nodes(dim_in, node1, node2)
    model = link_nodes(dim_in, model, node3)
    return model


def get_org_model(dim_in, dim_out, hidden):
    activation = SigmoidActivation()
    input_gate = Node(1, dim_in, dim_out, activation)
    hidden_layer = Node(1, hidden, hidden, activation)
    output_gate = Node(1, hidden, dim_out, activation)
    model = Organism([input_gate])
    model.set_optimization_risk(True)
    return model


def main_link_model(dataset='xor'):
    dim_in = 3
    dim_mid = 10
    dim_out = 1
    arity = 1

    activation_function = SigmoidActivation()

    model = get_link_model(activation_function, arity, dim_in, dim_mid,
                           dim_out)
    launch_model(model, data=dataset)


def main_org_model():
    dim_in = 2
    dim_mid = 3
    dim_out = 1

    model = get_org_model(dim_in, dim_out, dim_mid)
    launch_model(model, max_iter=100, data='dummy')


if __name__ == "__main__":
    main_link_model('iris')
    # main_org_model()
