import numpy as np

from core.cell.operands.operand import Operand
from core.cell.optim.loss import get_predicted
from core.cell.optim.optimization_args import OptimizationArgs


class Optimization:
    def __init__(self, cell, optim_args: OptimizationArgs,
                 decay_rate=0.99999, risk=False):
        """
        Initialize the Optimization object.

        :param cell: The model or neural network cell to be optimized.
        :param decay_rate: The rate at which the learning rate decays per
        iteration.
        """
        self.risk = risk
        self.cell = cell
        self.fit_func = optim_args.fitness_function
        self.input = optim_args.inputs
        self.desired_output = optim_args.desired_output
        self.max_iter = optim_args.max_iter
        self.decay_rate = decay_rate
        self.weights = cell.get_weights()
        self.prev_weights = [weight.get() for weight in self.weights]
        self.prev_error = cell.error
        self.learning_rate = optim_args.learning_rate
        self._initial_learning_rate = optim_args.learning_rate
        self._vanishing = set()
        self._exploding = set()

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
        Calculate the gradients of the fitness function with respect to the
        weights.

        :return: A numpy array of gradients.
        """
        gradients = np.array([
            self.fit_func.gradient(
                self.cell, self.input, self.desired_output, j
            )
            for j in range(len(self.weights))
        ])
        # print(gradients)
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
        weight.set(weight.get() - self.learning_rate * gradient)
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


class Optimizer:

    def __init__(self):
        self.risk = False

    def __call__(self, cell: Operand,
                 desired_output,
                 fit_fct,
                 learning_rate,
                 max_iterations,
                 variables):
        """
        Optimize a cell

        :param cell: The model or neural network cell to be optimized.
        :param fit_fct: The fitness function used to evaluate the model.
        :param variables: The input variables for the model.
        :param desired_output: The desired output for the input variables.
        :param max_iterations: The maximum number of iterations for optimization.
        :param learning_rate: The initial learning rate for gradient descent.
        """
        optim_args = OptimizationArgs()
        optim_args.desired_output = desired_output
        optim_args.fitness_function = fit_fct
        optim_args.learning_rate = learning_rate
        optim_args.max_iter = max_iterations
        optim_args.inputs = variables
        optimizer = Optimization(cell, optim_args, self.risk)
        # print("Under Optimization:", cell.id)
        optimizer.optimize()
