import sys

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from core.cell import Linker, MSEMultivariate, OptimizationArgs, Optimizer
from core.organism.activation_function import SigmoidActivation
from core.organism.node import Node


def load_iris_dataset(filename):
    df = pd.read_csv(filename)
    species_mapping = {'setosa': 0, 'versicolor': 0.5, 'virginica': 1}
    df['species'] = df['species'].map(species_mapping)
    X = df.drop('species', axis=1).values
    y = df['species'].values.reshape(-1, 1)
    return train_test_split(X, y, test_size=0.2, random_state=42)


def create_batches(X, y, batch_size):
    for start in range(0, len(X), batch_size):
        end = min(start + batch_size, len(X))
        yield X[start:end], y[start:end]


def get_model(dim_in, dim_mid, dim_out, hidden_layer_count, activation_function):
    input_node = Node(1, dim_in, dim_mid, activation_function, Optimizer())
    output_node = Node(1, dim_mid, dim_out, activation_function, Optimizer())
    model = input_node
    for i in range(hidden_layer_count):
        hl = Node(1, dim_mid, dim_mid, activation_function, Optimizer())
        model = Linker(hl, model, dim_in)
    model = Linker(output_node, model, dim_out)
    model.set_optimization_risk(True)
    return model


def main():
    dim_in = 4
    dim_mid = 10
    dim_out = 1
    arity = 1
    hiddens = 1

    activation_function = SigmoidActivation()

    model = get_model(dim_in, dim_mid, dim_out, hiddens, activation_function)

    X_train, X_test, y_train, y_test = load_iris_dataset('iris_dataset.csv')
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    batch_size = 5
    epochs = 275

    loss = MSEMultivariate()
    learning_rate = 0.1
    decay = 1e-10
    for epoch in range(epochs):
        epoch_loss = 0
        for X_batch, y_batch in create_batches(X_train, y_train, batch_size):
            input_data = [[np.array(x)] for x in X_batch]
            desired_output = [[np.array([y[0]])] for y in y_batch]

            optimization_args = OptimizationArgs(
                inputs=input_data,
                desired_output=desired_output,
                loss_function=loss,
                learning_rate=learning_rate,
                max_iter=1,  # Iteration within a batch
                min_error=sys.maxsize
            )
            epoch_loss += train(model, optimization_args)
        learning_rate -= decay
        epoch_loss /= len(X_train) // batch_size
        accuracy = evaluate_accuracy(model, X_test, y_test)

        print(f"Epoch {epoch + 1}/{epochs} - Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}")

    # Final evaluation on test set
    test_loss, test_accuracy = evaluate(model, X_test, y_test)
    print(f"Test set performance - Loss: {test_loss:.4f}, Accuracy: {test_accuracy:.4f}")


def train(model, optimization_args):
    model.optimize(optimization_args)
    return optimization_args.loss_function.evaluate(model, optimization_args.inputs, optimization_args.desired_output)


def evaluate(model, X, y):
    input_data = [[np.array(x)] for x in X]
    desired_output = [[np.array([y_i[0]])] for y_i in y]
    loss_function = MSEMultivariate()
    loss = loss_function.evaluate(model, input_data, desired_output)
    accuracy = evaluate_accuracy(model, X, y)
    return loss, accuracy


def evaluate_accuracy(model, X, y):
    input_data = [[np.array(x)] for x in X]
    predictions = np.array([np.round(model(x)[0] * 2) for x in input_data])
    predictions = predictions * 0.5  # Rounding to nearest 0, 0.5, or 1
    correct_predictions = np.sum(predictions == y.flatten())
    accuracy = correct_predictions / len(y)
    return accuracy


if __name__ == "__main__":
    main()
