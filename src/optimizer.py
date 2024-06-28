import tensorflow as tf


class Optimizer:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_values(self, variables, gradient):
        # Update the values of the "value" nodes using gradient descent
        for var, grad in zip(variables, gradient):
            var.assign_sub(self.learning_rate * grad)


def loss_function(output, desired_output):
    # Mean squared error loss
    return tf.reduce_mean(tf.square(output - desired_output))
