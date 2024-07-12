import numpy as np

from core.cell.operands.operand import Operand
from core.cell.optim.loss import get_predicted
from core.cell.optim.optimization_args import OptimizationArgs


class Optimization:
    def __init__(
            self, cell, optim_args: OptimizationArgs,
            decay_rate=0.99999, risk=False, ewc_lambda=0.1, l2_lambda=0.0
    ):
        """
        Initialize the Optimization object.

        :param cell: The model or neural network cell to be optimized.
        :param decay_rate: The rate at which the learning rate decays per
        iteration.
        :param ewc_lambda: The regularization strength for EWC.
        :param l2_lambda: The regularization strength for L2 regularization.
        """
        self.risk = risk
        self.cell = cell
        self.fit_func = optim_args.fitness_function
        self.input = optim_args.inputs
        self.desired_output = optim_args.desired_output
        self.max_iter = optim_args.max_iter
        self.decay_rate = decay_rate
        self.ewc_lambda = ewc_lambda
        self.l2_lambda = l2_lambda
        self.weights = cell.get_weights()
        self.prev_weights = [weight.get() for weight in self.weights]
        self.prev_error = cell.error
        self.learning_rate = optim_args.learning_rate
        self._initial_learning_rate = optim_args.learning_rate
        self._vanishing = set()
        self._exploding = set()
        # Initialize EWC importance
        self.ewc_importance = np.ones(len(self.weights))

    def optimize(self):
        """
        Perform optimization over the specified number of iterations.
        """
        for iteration in range(self.max_iter):
            gradients = self.calculate_gradients()
            self.update_weights(gradients)
            self.cell.error = self.fit_func(
                self.desired_output,
                get_predicted(self.input, self.cell)
            )
            if self.cell.error < 1e-20:
                break

    def calculate_gradients(self):
        """
        Calculate the gradients of the fitness function with respect to the weights.

        :return: A numpy array of gradients.
        """
        gradients = np.array([
            self.fit_func.gradient(
                self.cell, self.input, self.desired_output, j
            )
            for j in range(len(self.weights))
        ])
        return gradients

    def update_weights(self, gradients):
        """
        Update the weights based on the calculated gradients.

        :param gradients: The calculated gradients for each weight.
        """
        for i, weight in enumerate(self.weights):
            if weight.is_locked:
                continue
            gradient = gradients[i]
            gradient = self.__handle_exploding_vanishing(gradient, i)

            self.__update_weight(gradient, i, weight)

    def __update_weight(self, gradient, i, weight):
        """
        Update a single weight and handle fitness evaluation and convergence
        checks.

        :param gradient: The gradient for the weight.
        :param i: The index of the weight.
        :param weight: The weight object.
        :return: True if the update was successful, False otherwise.
        """
        # Regularization term for EWC
        ewc_term = self.ewc_lambda * self.ewc_importance[i] * (
                weight.get() - self.prev_weights[i]
        )
        # Regularization term for L2
        l2_term = self.l2_lambda * weight.get()

        weight.set(
            weight.get() - self.learning_rate * (gradient + ewc_term + l2_term)
        )
        y_pred = [self.cell(x_inst) for x_inst in self.input]
        self.cell.error = self.fit_func(self.desired_output, y_pred)
        self.learning_rate *= self.decay_rate
        if not self.risk:
            if self.prev_error < self.cell.error:
                self.cell.error = self.prev_error
                self.weights[i].set(self.prev_weights[i])
                return self.__test_vanishing_exploding(gradient, i, weight)
        return True

    def __test_vanishing_exploding(self, gradient, i, weight):
        """
        Test for vanishing or exploding gradients and take appropriate actions.

        :param gradient: The gradient for the weight.
        :param i: The index of the weight.
        :param weight: The weight object.
        """
        if i in self._vanishing or i in self._exploding:
            return False
        if np.abs(gradient) > 1:
            self.__mark_exploding(i)
        elif np.abs(gradient) < 1e-5:
            self.__mark_vanishing(i)
        else:
            return False
        gradient = self.__handle_exploding_vanishing(gradient, i)
        return self.__update_weight(gradient, i, weight)

    def __mark_exploding(self, i):
        """
        Mark the weight as having an exploding gradient.

        :param i: The index of the weight.
        """
        self._exploding.add(i)

    def __mark_vanishing(self, i):
        """
        Mark the weight as having a vanishing gradient.

        :param i: The index of the weight.
        """
        self._vanishing.add(i)

    def __handle_exploding_vanishing(self, gradient, i):
        if i in self._exploding:
            self.learning_rate = self._initial_learning_rate
            return np.clip(gradient, -1, 1)
        if i in self._vanishing:
            self.learning_rate = self._initial_learning_rate
            return np.sign(gradient) * np.maximum(np.abs(gradient), 1e-4)
        return gradient

    def update_ewc_importance(self, fisher_information):
        """
        Update the EWC importance based on the Fisher Information Matrix.

        :param fisher_information: The Fisher Information Matrix.
        """
        self.ewc_importance = fisher_information

    def ewc_loss(self, new_weights):
        """
        Calculate the EWC loss based on the changes in important weights.

        :param new_weights: The updated weights after an optimization step.
        :return: The EWC loss term.
        """
        ewc_loss = 0.0
        for old_weight, new_weight, importance in zip(self.prev_weights,
                                                      new_weights,
                                                      self.ewc_importance):
            ewc_loss += self.ewc_lambda * importance * (
                        new_weight - old_weight) ** 2
        return ewc_loss


class Optimizer:

    def __init__(self):
        self.risk = False

    def __call__(self, cell: Operand,
                 args: OptimizationArgs):
        """
        Optimize a cell

        :param cell: The model or neural network cell to be optimized.
        :param args: The arguments used in the optimization.
        """
        optimizer = self.make_optimizer(cell, args)
        # print("Under Optimization:", cell.id)
        optimizer.optimize()

    def make_optimizer(self, cell, optim_args, ewc_lambda=0.0,
                       l2_lambda=0.0):
        optimizer = Optimization(cell, optim_args, self.risk,
                                 ewc_lambda=ewc_lambda, l2_lambda=l2_lambda)
        return optimizer


def calculate_fisher_information(cell, inputs, old_fisher):
    """
    Calculate the Fisher Information Matrix.

    :param cell: The model or neural network cell.
    :param inputs: The input data.
    :param old_fisher: The Old Fisher Information Matrix.
    :return: The New Fisher Information Matrix.
    """
    fisher_information = old_fisher

    # TODO
    for i, weight in enumerate(cell.get_weights()):
        grad = np.mean(np.array(cell.derive(i)(inputs)))
        # derive
        # method
        fisher_information[i] += np.square(grad)

    return fisher_information / len(inputs)


class FisherOptimizer(Optimizer):

    def __init__(self):
        super().__init__()
        self.risk = True
        self.fisher_information = None

    def __call__(self, cell: Operand, opt: OptimizationArgs):
        """
        Optimize a cell

        :param cell: The model or neural network cell to be optimized.
        :param args: The arguments used in the optimization.
        """
        optimizer = self.make_optimizer(
            cell, opt, ewc_lambda=0.1, l2_lambda=0.01
        )
        if self.fisher_information is None:
            self.fisher_information = np.ones(len(cell.get_weights()))
        else:
            optimizer.update_ewc_importance(self.fisher_information)
        optimizer.optimize()
        self.fisher_information = calculate_fisher_information(
            cell, opt.inputs, self.fisher_information
        )
