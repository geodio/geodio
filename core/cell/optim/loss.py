import sys
from abc import ABC, abstractmethod

import numpy as np

from core.cell.operands.operand import Operand
from core.cell.operands.stateful import Stateful
from core.utils import flatten


def get_predicted(X, cell):
    return flatten([cell(x_inst) for x_inst in X])


class LossFunction(ABC):

    def evaluate(self, cell: Operand, X, Y):
        predicted = get_predicted(X, cell)
        return self.compute_fitness(flatten(Y), flatten(predicted))

    @abstractmethod
    def compute_fitness(self, Y, predicted):
        pass

    @abstractmethod
    def compute_d_fitness(self, Y, predicted):
        pass

    def __call__(self, Y, predicted):
        return self.compute_fitness(flatten(Y), flatten(predicted))

    @abstractmethod
    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        pass

    def get_y_minus_predicted(self, Y, predicted):
        Y_minus_predicted = np.array(Y) - np.array(predicted[:len(Y)])
        return Y_minus_predicted


class MSE(LossFunction):
    def compute_fitness(self, Y, predicted):
        # Mean Squared Error (MSE) fitness function
        x = np.mean(self.get_y_minus_predicted(Y, predicted) ** 2)
        if str(x) == "nan" or str(x) == 'inf':
            x = sys.maxsize / 2
        return x

    def compute_d_fitness(self, Y, predicted):
        return predicted - Y

    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        predicted = get_predicted(X, cell)
        delta_f_w_j = cell.derive(index, by_weight)
        gradient_results = np.array([delta_f_w_j(X_i) for X_i in X])
        result = self.compute_gradient(Y, gradient_results, predicted)
        return result

    def compute_gradient(self, Y, gradient_results, predicted):
        Y = flatten(Y)
        predicted = flatten(predicted)
        gradient_results = flatten(gradient_results)[:len(Y)]
        per_i = - self.get_y_minus_predicted(Y, predicted) * gradient_results
        result = 2 * np.mean(per_i)
        if str(result) == "nan" or str(
                result) == 'inf' or result == np.inf:
            result = 0.0
        return result


class CheckpointedMSE(MSE):

    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        assert isinstance(cell, Stateful) and isinstance(cell, Operand)
        predicted = get_predicted(X[0], cell)
        delta_f_w_j = cell.derive(index, by_weight)
        gradient_results = [delta_f_w_j(X_i) for X_i in X[0]]

        if len(X) > 1:
            cell.use_checkpoint()
            predicted.extend(get_predicted(X[1], cell))
            gradient_results.extend([delta_f_w_j(X_i) for X_i in X[1]])
            gradient_results = np.array(gradient_results)
            cell.use_current()

        # print("LOSS_CHECKPOINT_MSE_GRADIENT", X, Y, predicted)
        result = self.compute_gradient(Y, gradient_results, predicted)
        return result


class MSEMultivariate(MSE):
    def compute_fitness(self, Y, predicted):
        Y = np.array(Y)
        predicted = np.array(predicted)
        squared_diff = (Y - predicted) ** 2
        mse = np.mean(squared_diff)
        if np.isnan(mse) or np.isinf(mse):
            mse = sys.maxsize / 2
        return mse

    def gradient(self, cell: Operand, X, Y, index, by_weight=True):
        # Flatten the nested input_data and desired_output for processing
        X_flat = [x[0] for x in X]
        Y_flat = [y[0] for y in Y]

        predicted = get_predicted(X, cell)
        delta_f_w_j = cell.derive(index, by_weight)
        jacobian_results = np.array([delta_f_w_j(x_i) for x_i in X])

        result = self.compute_multivariate_gradient(Y_flat, jacobian_results, predicted, len(X_flat))
        return result

    def compute_multivariate_gradient(self, Y, jacobian_results, predicted, train_sets):
        Y = np.array(Y)
        predicted = np.array(predicted)
        jacobian_results = np.array(jacobian_results)

        predicted = predicted.reshape(Y.shape)
        diff = Y - predicted

        # Print shapes for debugging
        # print("PREDICTED", predicted.shape)
        # print("Y", Y.shape)
        # print("JACOBIAN", jacobian_results.shape)
        # print("DIFF", diff.shape)
        if jacobian_results.ndim == 3 and diff.ndim == 3:
            per_instance_grad = -2 * np.einsum('ijk,ik->ij', diff,
                                               jacobian_results)
        elif jacobian_results.ndim == 2 and diff.ndim == 2:
            per_instance_grad = -2 * np.matmul(diff.T,
                                               jacobian_results)
        else:
            diff = diff[:, :, np.newaxis]
            per_instance_grad = -2 * diff * jacobian_results
        gradient = np.mean(per_instance_grad, axis=0)

        if np.isnan(gradient).any() or np.isinf(gradient).any():
            gradient = np.zeros_like(gradient)
        return gradient
