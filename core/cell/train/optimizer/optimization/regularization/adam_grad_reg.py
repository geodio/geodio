import numpy as np

from core.cell.train.optimizer.utils import adapt_gradient
from core.cell.operands.weight import t_weight
from core.cell.train.optimizer.optimization.regularization.base_grad_reg \
    import (BaseGradReg)


class AdamGradReg(BaseGradReg):
    def __init__(self, weight: t_weight, optim_args):
        """
        Initialize the Adam gradient regularizer.

        :param weight: The weight to be regularized.
        :param optim_args: Optimization arguments including decay rate, L2,
        EWC lambda, etc.
        """
        super().__init__(weight, optim_args)
        self.m = np.zeros(weight.get().shape)  # Initialize first moment vector
        self.v = np.zeros(weight.get().shape)  # Initialize second moment
        # vector
        self.t = 0  # Initialize timestep

        # Adam-specific hyperparameters
        self.beta1 = optim_args.beta1 if hasattr(optim_args, 'beta1') else 0.9
        self.beta2 = optim_args.beta2 if hasattr(optim_args,
                                                 'beta2') else 0.999
        self.epsilon = optim_args.epsilon if hasattr(optim_args,
                                                     'epsilon') else 1e-8

    def apply_regularization(self, gradient, cell):
        """
        Apply Adam regularization along with L2, EWC, and vanishing/exploding gradient handling.

        :param gradient: The calculated gradient to be regularized.
        :param cell: The computational unit (e.g., neural network).
        """
        # Increment timestep
        self.t += 1

        # Apply EWC and L2 regularization
        ewc_term = self.optim_args.ewc_lambda * self.ewc_importance * (
                    self.weight.get() - self.prev_weight)
        l2_term = self.optim_args.l2_lambda * self.weight.get()
        gradient = adapt_gradient(gradient, self.weight)
        total_gradient = gradient + ewc_term + l2_term

        # Adam update rules
        # Update biased first moment estimate
        self.m = self.beta1 * self.m + (1 - self.beta1) * total_gradient
        # Update biased second moment estimate
        self.v = self.beta2 * self.v + (1 - self.beta2) * (
                    total_gradient ** 2)

        # Correct bias in first and second moments
        m_hat = self.m / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        # Compute the update
        niu = self.optim_args.learning_rate / self.optim_args.batch_size
        corrected_gradient = m_hat / (np.sqrt(v_hat) + self.epsilon)

        # Handle vanishing/exploding gradients
        current_error = self.correct_and_verify(cell, niu, corrected_gradient)
        return current_error
