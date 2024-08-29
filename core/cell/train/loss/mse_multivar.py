import sys
from typing import List

import numpy as np

from core.cell.operands.operand import Operand
from core.cell.train.forest import forest_derive
from core.cell.operands.utility import get_predicted
from core.cell.train.loss.mse import MSE


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

    def multi_gradient(self, cell, inputs, desired_outputs,
                       operands: List[Operand]):
        m_tree = forest_derive(cell, operands)
        Y_flat = [y[0] for y in desired_outputs]
        predicted = get_predicted(inputs, cell)
        m_jacobian_results = [m_tree(np.array(x_i)) for x_i in inputs]

        transposed_tuples = list(zip(*m_jacobian_results))
        m_jacobian_results = [list(sublist) for sublist in transposed_tuples]

        return [
            self.compute_multivariate_gradient(Y_flat,
                                               jacobian_results,
                                               predicted)
            for jacobian_results in m_jacobian_results
        ]

    def compute_multivariate_gradient(self, Y, jacobian_results, predicted):
        Y = np.array(Y)
        predicted = np.array(predicted)
        jacobian_results = np.array(jacobian_results)

        predicted = predicted.reshape(Y.shape)
        diff = Y - predicted
        try:
            if jacobian_results.ndim == 3 and diff.ndim == 3:
                per_instance_grad = -2 * np.einsum('ijk,ik->ij', diff,
                                                   jacobian_results)
            elif jacobian_results.ndim == 2 and diff.ndim == 2:
                per_instance_grad = -2 * np.matmul(diff.T,
                                                   jacobian_results)
            else:
                diff = diff[:, :, np.newaxis]
                per_instance_grad = -2 * diff * jacobian_results
        except:
            if jacobian_results.ndim == 4:
                try:
                    diff = diff[:, :, :, np.newaxis]
                    per_instance_grad = -2 * diff * jacobian_results
                    per_instance_grad = np.sum(per_instance_grad, axis=(1))
                except:
                    diff = np.array([(Y - predicted).T])
                    diff = diff[:, :, np.newaxis, np.newaxis]
                    per_instance_grad = -2 * diff * jacobian_results
                    per_instance_grad = np.sum(per_instance_grad, axis=(1))
            else:
                jacobian_results = np.transpose(jacobian_results,
                                                axes=(0, 2, 1))
                per_instance_grad = -2 * diff * jacobian_results
                per_instance_grad = np.mean(per_instance_grad, axis=(1))
        gradient = np.mean(per_instance_grad, axis=0)

        if np.isnan(gradient).any() or np.isinf(gradient).any():
            gradient = np.zeros_like(gradient)
        return gradient
