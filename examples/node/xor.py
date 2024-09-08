import sys
import numpy as np

from core.cell import Linker, b_var
from core.cell import MSEMultivariate
from core.cell import OptimizationArgs
from core.cell import Optimizer
from core.organism.activation_function import SigmoidActivation
from core.organism.node import Node
from core.organism.organism import Organism


def main(dataset_name=None, optimizer=None):
    dataset_name = dataset_name or 'xor'
    dataset = get_dataset(dataset_name)
    model_config = get_model_config(dataset_name)

    model = create_model(model_config)
    if optimizer:
        model.optimizer = optimizer
    train_model(model, dataset, model_config)


def get_dataset(name):
    datasets = {
        'xor': xor_data,
        'xor_2': xor_2_data,
        'xor_3': xor_3_data,
        'iris': iris_sepal,
        'dummy': dummy_data,
    }
    if name not in datasets:
        raise ValueError(f"Unknown dataset: {name}")
    return datasets[name]()


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


def xor_3_data():
    input_data = [
        [np.array([1, 1, 0])],
        [np.array([0, 1, 0])],
        [np.array([1, 0, 0])],
        [np.array([0, 0, 0])]
    ]
    desired_output = [
        [np.array([1.0, 0.0])],
        [np.array([0.0, 1.0])],
        [np.array([0.0, 1.0])],
        [np.array([1.0, 0.0])]
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
        [np.array([0, 1, 0])],
        [np.array([0, 1, 0])],
        [np.array([1, 0, 0])],
        [np.array([0, 0, 1])],
        [np.array([0, 0, 1])],
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


def train_model(model, dataset, model_config):
    desired_output, input_data = dataset
    loss_function = MSEMultivariate()

    optimization_args = OptimizationArgs(
        inputs=input_data,
        desired_output=desired_output,
        loss_function=loss_function,
        learning_rate=0.1,
        max_iter=model_config.get('max_iter', 10000),
        min_error=sys.maxsize
    )
    train(desired_output, input_data, model, loss_function, optimization_args)


def train(desired_output, input_data, model, loss_function, optimization_args):
    output_before = [model(input_data_i) for input_data_i in input_data]
    print("Weights before optimization:")
    print([[w, w.w_index] for w in model.get_weights()])
    print(loss_function.evaluate(model, input_data, desired_output))
    model.optimize(optimization_args)
    print("Weights after optimization:")
    print([[w, w.w_index] for w in model.get_weights()])
    print(loss_function.evaluate(model, input_data, desired_output))
    output = [model(input_data_i) for input_data_i in input_data]
    print("            Input         :", input_data)
    print("Before Optimization output:", output_before)
    print("       Final output       :", output)
    print("       Desired output     :", desired_output)


def get_model_config(dataset_name):
    model_configs = {
        'xor': {'dim_in': 2, 'dim_mid': 10, 'dim_out': 1, 'arity': 1},
        'xor_2': {'dim_in': 2, 'dim_mid': 10, 'dim_out': 1, 'arity': 1},
        'xor_3': {'dim_in': 3, 'dim_mid': 10, 'dim_out': 2, 'arity': 1},
        'iris': {'dim_in': 3, 'dim_mid': 10, 'dim_out': 3, 'arity': 1},
        'dummy': {'dim_in': 2, 'dim_mid': 3, 'dim_out': 1, 'arity': 1,
                  'max_iter': 100},
    }
    if dataset_name not in model_configs:
        raise ValueError(f"Unknown model config for dataset: {dataset_name}")
    return model_configs[dataset_name]


def create_model(config):
    activation_function = SigmoidActivation([b_var()])
    arity = config['arity']
    dim_in = config['dim_in']
    dim_mid = config['dim_mid']
    dim_out = config['dim_out']

    # model = Organism.create_simple_organism(
    #     dim_in=dim_in,
    #     dim_hidden=dim_mid,
    #     hidden_count=1,
    #     dim_out=dim_out,
    #     activation_function=activation_function,
    #     optimizer=Optimizer()
    # )
    node2 = make_node(activation_function.clone(), arity, dim_mid, dim_out)
    node1 = make_node(activation_function.clone(), arity, dim_in, dim_mid)
    model = link_nodes(dim_in, node1, node2)
    return model


def link_nodes(dim_in, node1, node2):
    leenk = Linker(node2, node1, dim_in, True)
    leenk.set_optimization_risk(True)
    return leenk


def make_node(activation_function, arity, dim_in, dim_out):
    node = Node(arity, dim_in, dim_out, activation_function,
                optimizer=Optimizer())
    node.set_optimization_risk(True)
    return node

# TODO
if __name__ == "__main__":
    main()
