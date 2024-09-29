import sys

import numpy as np

from geodio.core.cell.operands.operand import Operand
from geodio.core.cell.operands.utility import get_predicted
from geodio.core.cell.train.loss.mse import MSE


class MSEMultivariate(MSE):
    def compute_fitness(self, desired_output, predicted):
        desired_output = np.array(desired_output)
        predicted = np.array(predicted)
        squared_diff = (desired_output - predicted) ** 2
        mse = np.mean(squared_diff)
        if np.isnan(mse) or np.isinf(mse):
            mse = sys.maxsize / 2
        return mse

    def gradient(self, cell: Operand, inputs, desired_output, index, by_weight=True):
        # Flatten the nested input_data and desired_output for processing
        Y_flat = [y[0] for y in desired_output]

        predicted = get_predicted(inputs, cell)
        delta_f_w_j = cell.derive(index, by_weight)
        jacobian_results = np.array([delta_f_w_j(x_i) for x_i in inputs])

        result = self.compute_multivariate_gradient(Y_flat, jacobian_results,
                                                    predicted)
        return result

    def compute_multivariate_gradient(self, Y, jacobian_results, predicted):
        Y = np.array(Y)
        predicted = np.array(predicted)
        jacobian_results = np.array(jacobian_results)

        predicted = predicted.reshape(Y.shape)
        diff = Y - predicted
        per_instance_grad = -2 * diff[:, :, np.newaxis] * jacobian_results
        gradient = np.mean(per_instance_grad, axis=0)

        if np.isnan(gradient).any() or np.isinf(gradient).any():
            gradient = np.zeros_like(gradient)
        return gradient
