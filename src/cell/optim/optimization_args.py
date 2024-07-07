class OptimizationArgs:
    def __init__(self,
                 learning_rate=0.01,
                 max_iter=1000,
                 min_fitness=10,
                 fitness_function=None,
                 inputs=0,
                 desired_output=0,
                 actual_output=0
                 ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.min_fitness = min_fitness
        self.fitness_function = fitness_function
        self.inputs = inputs
        self.desired_output = desired_output
        self.actual_output = actual_output
