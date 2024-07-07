import numpy as np

from src.cell.operands.operand import Operand
from src.cell.optim.fitness import get_predicted


class Optimization:
    def __init__(self, cell, fit_func, input_vars, desired_output, max_iter,
                 learning_rate, decay_rate=0.99999):
        """
        Initialize the Optimization object.

        :param cell: The model or neural network cell to be optimized.
        :param fit_func: The fitness function used to evaluate the model.
        :param input_vars: The input variables for the model.
        :param desired_output: The desired output for the input variables.
        :param max_iter: The maximum number of iterations for optimization.
        :param learning_rate: The initial learning rate for gradient descent.
        :param decay_rate: The rate at which the learning rate decays per iteration.
        """
        self.cell = cell
        self.fit_func = fit_func
        self.input = input_vars
        self.desired_output = desired_output
        self.max_iter = max_iter
        self.decay_rate = decay_rate
        self.weights = cell.get_weights()
        self.prev_weights = [weight.get() for weight in self.weights]
        self.prev_fitness = cell.fitness
        self.learning_rate = learning_rate
        self._initial_learning_rate = learning_rate
        self._vanishing = set()
        self._exploding = set()

    def optimize(self):
        """
        Perform optimization over the specified number of iterations.
        """
        for iteration in range(self.max_iter):
            gradients = self.calculate_gradients()
            self.update_weights(gradients)
            self.cell.fitness = self.fit_func(
                self.desired_output,
                get_predicted(self.input, self.cell)
            )
            if self.cell.fitness < 1e-20:
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
            gradient = gradients[i]
            gradient = self.__handle_exploding_vanishing(gradient, i)

            self.__update_weight(gradient, i, weight)

    def __update_weight(self, gradient, i, weight):
        """
        Update a single weight and handle fitness evaluation and convergence checks.

        :param gradient: The gradient for the weight.
        :param i: The index of the weight.
        :param weight: The weight object.
        :return: True if the update was successful, False otherwise.
        """
        weight.set(weight.get() - self.learning_rate * gradient)
        y_pred = [self.cell(x_inst) for x_inst in self.input]
        self.cell.fitness = self.fit_func(self.desired_output, y_pred)
        self.learning_rate *= self.decay_rate

        # if self.prev_fitness < self.cell.fitness:
        #     self.cell.fitness = self.prev_fitness
        #     self.weights[i].set(self.prev_weights[i])
        #     return self.__test_vanishing_exploding(gradient, i, weight)
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

    def __call__(self, cell: Operand,
                 desired_output,
                 fit_fct,
                 learning_rate,
                 max_iterations,
                 variables):
        optimizer = Optimization(cell,
                                 fit_fct,
                                 variables,
                                 desired_output,
                                 max_iterations,
                                 learning_rate)
        print("OPTIMIZER <<<<", variables, desired_output, ">>>>")
        optimizer.optimize()
