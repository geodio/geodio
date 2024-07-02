import numpy as np

from src.cell.operands.operand import Operand


class Optimization:
    def __init__(self, cell, fit_func, input_vars, desired_output, max_iter,
                 learning_rate, decay_rate=0.99999):
        self.cell = cell
        self.fit_func = fit_func
        self.input = input_vars
        self.desired_output = desired_output
        self.max_iter = max_iter
        self.decay_rate = decay_rate
        self.weights = cell.get_weights()
        self.prev_weights = [weight.weight for weight in self.weights]
        self.prev_fitness = cell.fitness
        self.learning_rate = learning_rate

    def optimize(self):
        for iteration in range(self.max_iter):
            gradients = self.calculate_gradients()
            self.update_weights(gradients)

    def calculate_gradients(self):
        gradients = np.array([
            self.fit_func.gradient(
                self.cell, self.input, self.desired_output, j
            )
            for j in range(len(self.weights))
        ])
        return gradients

    def update_weights(self, gradients):
        for i, weight in enumerate(self.weights):
            gradient = gradients[i]
            self.__update_weight(gradient, i, weight)

    def __update_weight(self, gradient, i, weight):
        weight.weight = weight.weight - self.learning_rate * gradient
        # Recalculate fitness
        y_pred = [self.cell(x_inst) for x_inst in self.input]
        self.cell.fitness = self.fit_func(self.desired_output, y_pred)
        self.learning_rate *= self.decay_rate
        # Convergence check
        if self.prev_fitness < self.cell.fitness:
            self.cell.fitness = self.prev_fitness
            self.weights[i].weight = self.prev_weights[i]
            self.__test_vanishing_exploding(gradient, i, weight)
            return False
        return True

    def __test_vanishing_exploding(self, gradient, i, weight):
        if gradient > 1:
            try_exploding = self.__update_weight(0.1, i + 1, weight)
            if try_exploding:
                self.__mark_exploding(i)
        if gradient < 1:
            try_vanishing = self.__update_weight(0.1, i + 1, weight)
            if try_vanishing:
                self.__mark_vanishing(i)

    def __mark_exploding(self, i):
        pass

    def __mark_vanishing(self, i):
        pass


class Optimizer:
    def __init__(self):
        pass

    def __call__(self, cell: Operand,
                 desired_output,
                 fit_fct,
                 learning_rate,
                 max_iterations,
                 variables):
        decay_rate = 0.99999
        weights = cell.get_weights()
        prev_weights = [weight.weight for weight in weights]
        prev_fitness = cell.fitness
        for iteration in range(max_iterations):
            gradients = self.calculate_gradient(cell, desired_output, fit_fct,
                                                variables, weights)
            np.clip(gradients, -1, 1, out=gradients)
            self.update_weights(cell, decay_rate, desired_output, fit_fct,
                                gradients, learning_rate, prev_fitness,
                                prev_weights, variables, weights)

    def calculate_gradient(self, cell, desired_output, fit_fct, variables,
                           weights):
        gradients = [fit_fct.gradient(cell, variables, desired_output, j)
                     for j in range(len(weights))]
        gradients = np.array(gradients)
        return gradients

    def update_weights(self, cell, decay_rate, desired_output, fit_fct,
                       gradients, learning_rate, prev_fitness, prev_weights,
                       variables, weights):
        for i, weight in enumerate(weights):
            weight.weight = weight.weight - learning_rate * gradients[i]
            # Recalculate fitness
            y_pred = [cell(x_inst) for x_inst in variables]
            cell.fitness = fit_fct(desired_output, y_pred)
            # Adjust learning rate based on age and mutation risk todo
            learning_rate *= decay_rate
            # Convergence check
            if prev_fitness < cell.fitness:
                cell.fitness = prev_fitness
                weights[i].weight = prev_weights[i]
