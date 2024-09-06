import sys

import numpy as np


def _get_shape(data):
    """Helper method to get the shape of inputs or desired_output."""
    if isinstance(data, list):
        data = np.array(data)
    return data.shape if data is not None else None


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
                 decay_rate=0,
                 **props
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
        self._merged_inputs = None
        self._merged_desired_output = None
        self.props = props
        if not props.get('backpropagation'):
            self.props['backpropagation'] = False
        if not props.get('optimizer_strategy'):
            self.props['optimizer_strategy'] = 'adam'

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
            decay_rate=self.decay_rate,
            **self.props.copy()
        )

    def compute_error(self):
        return self.loss_function(self.actual_output, self.desired_output)

    def batches(self):
        x = self.inputs
        y = self.desired_output
        for start in range(0, len(x), self.batch_size):
            end = min(start + self.batch_size, len(x))
            yield x[start:end], y[start:end]

    def _merge_inputs(self):
        if self._merged_inputs is None:
            self._merged_inputs = [
                np.array([x[0] for x in self.inputs]).T
            ]
        return self._merged_inputs

    def _merge_desired_outputs(self):
        if self._merged_desired_output is None:
            self._merged_desired_output = [
                np.array([x[0] for x in self.desired_output]).T
            ]
        return self._merged_desired_output

    merged_inputs = property(lambda self: self._merge_inputs())

    merged_desired_output = property(
        lambda self: self._merge_desired_outputs())

    @staticmethod
    def split_desired_output(merged_d_o):
        # print("SPDO)", np.shape(merged_d_o))
        c_0 = merged_d_o[0]
        c_1 = c_0.T
        c_2 = c_1.tolist()
        c_3 = [[x] for x in c_2]
        return c_3

    @staticmethod
    def split_inputs(merged_in):
        # print("SPIN)", np.shape(merged_in))
        c_0 = merged_in.T
        c_1 = c_0.tolist()
        c_2 = [[x] for x in c_1]
        return c_2

    def __str__(self):
        inputs_shape = _get_shape(self.inputs)
        desired_output_shape = _get_shape(self.desired_output)
        return (
            f"OptimizationArgs("
            f"learning_rate={self.learning_rate}, "
            f"max_iter={self.max_iter}, "
            f"loss_function={self.loss_function}, "
            f"inputs_shape={inputs_shape}, "
            f"desired_output_shape={desired_output_shape}, "
            f"actual_output={self.actual_output}, "
            f"min_error={self.min_error}, "
            f"batch_size={self.batch_size}, "
            f"epochs={self.epochs}, "
            f"scaler={self.scaler}, "
            f"decay_rate={self.decay_rate}"
            f")"
        )

    backpropagation = property(lambda self: self.props.get('backpropagation'))
    optimizer_strategy = property(
        lambda self: self.props.get['optimizer_strategy']
    )
