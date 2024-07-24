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
        self.fit_func = optim_args.loss_function
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
            self.calculate_gradient(j) for j in range(len(self.weights))
        ])
        return gradients

    def calculate_gradient(self, j):
        """
        Calculate the gradient of the fitness function with respect to one
        weight.

        :param j: The index of the weight.
        :return: The gradient for the weight with index j.
        """
        gradient = self.fit_func.gradient(
            self.cell, self.input, self.desired_output, j
        )
        return gradient

    def update_weights(self, gradients):
        """
        Update the weights based on the calculated gradients.

        :param gradients: The calculated gradients for each weight.
        """
        for i, weight in enumerate(self.weights):
            if weight.is_locked:
                continue
            gradient = gradients[i]
            self.update_weight(i, gradient)

    def update_weight(self, w_index, gradient):
        """
        Update the weight based on the calculated gradients.

        :param w_index: The index of the weight.
        :param gradient: The calculated gradient for the weight with index
        w_index.
        :return: None
        """
        gradient = self.__handle_exploding_vanishing(gradient, w_index)
        self.__update_weight(gradient, w_index, self.weights[w_index])

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
        # print("INSIDE OPTIMIZATION, __update_weight", weight, gradient)
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


class RollingOptimization(Optimization):
    def optimize(self):
        for w_index in range(len(self.weights)):
            self.learning_rate = self._initial_learning_rate
            if self.weights[w_index].is_locked:
                continue
            # print("OPTIMIZING WEIGHT WITH INDEX", w_index)
            for iteration in range(self.max_iter):
                gradient = self.calculate_gradient(w_index)
                self.update_weight(w_index, gradient)
                if self.cell.error < 1e-20:
                    break


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
        optim_args = optim_args.clone()
        optimizer = RollingOptimization(cell, optim_args, self.risk,
                                        ewc_lambda=ewc_lambda,
                                        l2_lambda=l2_lambda)
        return optimizer

    def clone(self):
        cloned = Optimizer()
        cloned.risk = self.risk
        return cloned


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
            cell, opt, ewc_lambda=0.05, l2_lambda=0.001
        )
        if self.fisher_information is None:
            self.fisher_information = np.ones(len(cell.get_weights()))
        else:
            optimizer.update_ewc_importance(self.fisher_information)
        optimizer.optimize()
        self.fisher_information = calculate_fisher_information(
            cell, opt.inputs, self.fisher_information
        )

    def clone(self):
        cloned = FisherOptimizer()
        cloned.risk = self.risk
        cloned.fisher_information = self.fisher_information
        return cloned
