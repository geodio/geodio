import numpy as np

from core.cell.train.optimizer.utils import adapt_gradient
from core.cell.operands.weight import t_weight


class BaseGradReg:
    def __init__(self, weight: t_weight, optim_args):
        """
        Initialize the gradient regularizer.

        :param weight: The weight to be regularized.
        :param optim_args: Optimization arguments including decay rate, L2,
        EWC lambda, etc.
        """
        self.weight = weight
        self.optim_args = optim_args
        self.prev_weight = weight.get()
        self.ewc_importance = np.ones(weight.shape)
        self.prev_error = float('inf')  # Initialize with a large value
        self.is_exploding = False
        self.is_vanishing = False

    def apply_regularization(self, gradient, cell):
        """
        Apply regularization methods (L2, EWC, vanishing/exploding gradients)
        on the gradient.
        Roll back if the error increases and risk is not allowed.

        :param gradient: The calculated gradient to be regularized.
        :param cell: The computational unit (e.g., neural network).
        """
        # print("WEIGHT", self.weight.get().shape)
        # print("GRADIENT", gradient.shape)
        # Apply EWC regularization
        ewc_term = self.optim_args.ewc_lambda * self.ewc_importance * (
                self.weight.get() - self.prev_weight
        )
        # Apply L2 regularization
        l2_term = self.optim_args.l2_lambda * self.weight.get()

        # Adapt gradient shape
        gradient = adapt_gradient(gradient, self.weight)

        # Combine gradient with regularization terms
        niu = self.optim_args.learning_rate / self.optim_args.batch_size
        updated_gradient = gradient + ewc_term + l2_term
        current_error = self.correct_and_verify(cell, niu, updated_gradient)
        return current_error

    def correct_and_verify(self, cell, niu, updated_gradient):
        corrected_gradient = self.handle_exploding_vanishing(updated_gradient)
        # Update weight
        new_weight = self.weight.get() - niu * corrected_gradient
        self.weight.set(new_weight)
        # Update learning rate with decay
        self.optim_args.learning_rate *= (1 - self.optim_args.decay_rate)
        y_pred = [cell(x_inst) for x_inst in self.optim_args.inputs]
        current_error = self.optim_args.loss_function(self.optim_args.
                                                      desired_output, y_pred)
        # Risk-based handling: Roll back if error increases and risk is not
        # allowed
        if not self.optim_args.risk and current_error > self.prev_error:
            # Roll back to the previous weight and error
            self.weight.set(self.prev_weight)
            current_error = self.prev_error
        else:
            # Update previous weight and error if the update is successful
            self.prev_weight = self.weight.get()
            self.prev_error = current_error
        return current_error

    def handle_exploding_vanishing(self, gradient):
        """
        Handle vanishing and exploding gradients by applying corrective
        measures.

        :param gradient: The gradient to check for vanishing/exploding
        behavior.
        :return: Corrected gradient if necessary.
        """
        if not self.optim_args.exp_van_correction:
            return gradient
        if (np.abs(gradient) > 1).any() or self.is_exploding:
            # Mark as exploding and clip the gradient
            self.is_exploding = True
            return np.clip(gradient, -1, 1)
        elif (np.abs(gradient) < 1e-5).any() or self.is_vanishing:
            # Mark as vanishing and adjust the gradient
            self.is_vanishing = True
            return np.sign(gradient) * np.maximum(np.abs(gradient), 1e-4)
        return gradient

