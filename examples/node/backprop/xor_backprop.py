import sys

from python.geodio.core import EpochedOptimizer, MSEMultivariate, OptimizationArgs
from examples.node.xor import create_model, get_model_config, get_dataset, \
    train


def main(dataset_name=None, optimizer=None):
    dataset_name = dataset_name or 'xor'
    dataset = get_dataset(dataset_name)
    model_config = get_model_config(dataset_name)

    model = create_model(model_config)
    if optimizer:
        model.optimizer = optimizer
    model.set_optimization_risk(True)
    train_model(model, dataset, model_config)


def train_model(model, dataset, model_config):
    desired_output, input_data = dataset
    loss_function = MSEMultivariate()

    optimization_args = OptimizationArgs(
        inputs=input_data,
        desired_output=desired_output,
        loss_function=loss_function,
        learning_rate=1,
        max_iter=1,
        min_error=sys.maxsize,
        batch_size=1,
        epochs=1000
    )
    train(desired_output, input_data, model, loss_function, optimization_args)


if __name__ == '__main__':
    main('xor_2', EpochedOptimizer())
