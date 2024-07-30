import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from core.cell.collections.builtin_functors import Linker
from core.cell.optim.loss import MSEMultivariate
from core.cell.optim.optimization_args import OptimizationArgs
from core.cell.optim.optimizer import Optimizer
from core.organism.activation_function import SigmoidActivation
from core.organism.link import Link
from core.organism.node import Node
from core.organism.organism import Organism


def reverse_one_hot(y):
    classes = np.unique(y)
    class_dict = {
        cls: i for i, cls in enumerate(classes)
    }
    class_weight = 1.0 / (len(classes) - 1)
    rev_classes = np.zeros((y.size, 1), dtype=int)
    for i, yi in enumerate(y):
        rev_classes[i] = class_weight * class_dict[yi]
    return rev_classes


def one_hot(y, classes):
    hot_y = []
    lcls = len(classes) - 1
    for i in range(len(y)):
        class_idx = int(y[i][0] * lcls)
        hot_y.append(classes[class_idx])
    return hot_y


def data_from_dict(t_X):
    keys = t_X.keys()
    columns = np.empty((len(keys), t_X.shape[0]))
    for i, key in enumerate(keys):
        columns[i] = t_X[key]
    columns = np.asarray(columns)
    return columns.T


def encapsulate(y):
    return [[x] for x in y]


def make_nodes(dim_in, dim_out, hidden):
    activation = SigmoidActivation()
    input_gate = Node(1, dim_in, hidden, activation)
    hidden_layer = Node(1, hidden, hidden, activation)
    output_gate = Node(1, hidden, dim_out, activation)
    model = Linker(1, hidden_layer, input_gate, dim_in)
    model = Linker(1, output_gate, model, dim_in)
    return model


def main():
    test_X, test_y, train_X, train_y, validation_X, validation_y = get_iris_dataset()
    classes = np.unique(train_y)
    test_X = encapsulate(data_from_dict(test_X))
    train_X = encapsulate(data_from_dict(train_X))
    validation_X = encapsulate(data_from_dict(validation_X))
    e_test_y = encapsulate(reverse_one_hot(test_y))
    e_train_y = encapsulate(reverse_one_hot(train_y))
    e_validation_y = encapsulate(reverse_one_hot(validation_y))
    dim_in = len(train_X[0][0])
    dim_out = 1
    hidden = 5

    model = make_nodes(dim_in, dim_out, hidden)

    loss = MSEMultivariate()

    optimization_args = OptimizationArgs(
        inputs=train_X,
        desired_output=e_train_y,
        loss_function=loss,
        learning_rate=0.1,
        max_iter=1000,
        min_error=sys.maxsize
    )
    starting_error = loss.evaluate(model, validation_X, e_validation_y)
    print("STARTING ERROR:", starting_error)
    get_accuracy_validation(classes, model, validation_X, validation_y)
    get_accuracy_training(classes, model, train_X, train_y)
    model.optimize(optimization_args)

    ending_error = loss.evaluate(model, validation_X, e_validation_y)
    print("ENDING ERROR:", ending_error)

    get_accuracy_validation(classes, model, validation_X, validation_y)
    get_accuracy_training(classes, model, train_X, train_y)


def get_accuracy_validation(classes, model, validation_X, validation_y):
    accuracy = get_accuracy(classes, model, validation_X, validation_y)
    print('\taccuracy using the validation dataset:\t{:.3f}'.format(accuracy))


def get_accuracy_training(classes, model, train_x, train_y):
    accuracy = get_accuracy(classes, model, train_x, train_y)
    print('\taccuracy using the training dataset:\t{:.3f}'.format(accuracy))


def get_accuracy(classes, model, train_x, train_y):
    prediction_y = [model(x) for x in train_x]
    hot_prediction = one_hot(prediction_y, classes)
    accuracy = accuracy_score(hot_prediction, train_y)
    return accuracy


def get_iris_dataset():
    iris_data = load_iris()
    iris = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
    # Add labels
    iris['Species'] = list(
        map(lambda i: iris_data.target_names[i], iris_data.target))
    sns.pairplot(iris, hue='Species')

    train, validation_test = train_test_split(iris, test_size=0.4)
    print("Train:\t\t", train.shape[0], "objects")
    validation, test = train_test_split(validation_test, test_size=.5)
    print("Validation:\t", validation.shape[0], "objects")
    print("Testing:\t", test.shape[0], "objects")
    train_X = train[
        ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
         'petal width (cm)']]
    train_y = train.Species
    validation_X = validation[
        ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
         'petal width (cm)']]
    validation_y = validation.Species
    test_X = test[
        ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)',
         'petal width (cm)']]
    test_y = test.Species
    return test_X, test_y, train_X, train_y, validation_X, validation_y


if __name__ == '__main__':
    main()
