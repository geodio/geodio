import sys


class OptimizationArgs:
    def __init__(self,
                 learning_rate=0.1,
                 max_iter=100,
                 loss_function=None,
                 inputs=0,
                 desired_output=0,
                 actual_output=0,
                 min_error=sys.float_info.max,
                 ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.inputs = inputs
        self.desired_output = desired_output
        self.actual_output = actual_output
        self.min_error = min_error

    def clone(self):
        return OptimizationArgs(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            loss_function=self.loss_function,
            inputs=self.inputs[:],
            desired_output=self.desired_output[:],
            actual_output=self.actual_output,
            min_error=self.min_error,
        )

    def compute_error(self):
        return self.loss_function(self.actual_output, self.desired_output)

    def get_weights(self):
        weights = []
        for child in self.get_sub_operands():
            weights.extend(child.get_weights())

        for i, weight in enumerate(weights):
            weight.w_index = i

        return weights

    def set_weights(self, new_weights):
        offset = 0
        for child in self.get_sub_operands():
            child_weights = child.get_weights()
            num_weights = len(child_weights)
            if num_weights > 0:
                child.set_weights(new_weights[offset:offset + num_weights])
                offset += num_weights

    def get_sub_operands(self):
        return []
