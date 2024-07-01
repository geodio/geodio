class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_values(self, variables, gradient):
        # Update the values of the "value" nodes using gradient descent
        for var, grad in zip(variables, gradient):
            var.assign_sub(self.learning_rate * grad)


