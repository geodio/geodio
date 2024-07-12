import sys


class OptimizationArgs:
    def __init__(self,
                 learning_rate=0.01,
                 max_iter=1000,
                 min_fitness=10,
                 fitness_function=None,
                 inputs=0,
                 desired_output=0,
                 actual_output=0,
                 min_error=sys.float_info.max,
                 ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_fitness = min_fitness
        self.fitness_function = fitness_function
        self.inputs = inputs
        self.desired_output = desired_output
        self.actual_output = actual_output
        self.min_error = min_error

    def clone(self):
        return OptimizationArgs(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            min_fitness=self.min_fitness,
            fitness_function=self.fitness_function,
            inputs=self.inputs[:],
            desired_output=self.desired_output[:],
            actual_output=self.actual_output,
            min_error=self.min_error,
        )

    def compute_error(self):
        return self.fitness_function(self.actual_output, self.desired_output)
