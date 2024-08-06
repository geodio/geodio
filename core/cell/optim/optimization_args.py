import sys


class OptimizationArgs:
    def __init__(self,
                 learning_rate=0.1,
                 max_iter=100,
                 loss_function=None,
                 inputs=None,
                 desired_output=None,
                 actual_output=None,
                 min_error=sys.float_info.max,
                 batch_size=1,
                 epochs=1,
                 scaler=None,
                 decay_rate=0
                 ):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.inputs = inputs
        self.desired_output = desired_output
        self.actual_output = actual_output
        self.min_error = min_error
        self.batch_size = batch_size
        self.epochs = epochs
        self.scaler = scaler
        self.decay_rate = decay_rate

    def clone(self):
        return OptimizationArgs(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            loss_function=self.loss_function,
            inputs=self.inputs[:],
            desired_output=self.desired_output[:],
            actual_output=self.actual_output,
            min_error=self.min_error,
            batch_size=self.batch_size,
            epochs=self.epochs,
            scaler=self.scaler,
            decay_rate=self.decay_rate
        )

    def compute_error(self):
        return self.loss_function(self.actual_output, self.desired_output)

    def batches(self):
        x = self.inputs
        y = self.desired_output
        for start in range(0, len(x), self.batch_size):
            end = min(start + self.batch_size, len(x))
            yield x[start:end], y[start:end]