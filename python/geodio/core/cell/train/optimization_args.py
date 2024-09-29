import sys
import numpy as np


def _get_shape(data):
    """
    Helper function to get the shape of a list or numpy array.

    Parameters:
    data (list or np.array): Input data to calculate the shape of.

    Returns:
    tuple: Shape of the data or None if data is None.
    """
    if isinstance(data, list):
        data = np.array(data)
    return data.shape if data is not None else None


class OptimizationArgs:
    """
    A class that stores and manages arguments for an optimization process.

    Attributes are stored dynamically in a dictionary `props`. This allows
    for flexible handling of parameters without needing to define explicit
    properties for every possible argument. Default values can be accessed
    via the `__getattr__` method.

    Parameters can be passed directly during initialization, allowing
    customization on a per-instance basis.
    """

    def __init__(self, learning_rate=0.1, max_iter=100, loss_function=None,
                 inputs=None, desired_output=None, **props):
        """
        Initialize the optimization arguments with dynamic properties.

        Parameters:
        learning_rate (float): The learning rate for the optimizer. Default is 0.1.
        max_iter (int): The maximum number of iterations. Default is 100.
        loss_function (callable): A callable function for computing the loss.
        inputs (array-like): Input data for the optimization process.
        desired_output (array-like): The expected output data.
        **props (dict): Additional parameters can be passed dynamically as keyword arguments.
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.loss_function = loss_function
        self.inputs = inputs
        self.desired_output = desired_output
        self._merged_inputs = None
        self._merged_desired_output = None
        self.props = props  # Dynamic properties storage

    def __getattr__(self, name):
        """
        Dynamically access attributes stored in `props` or fallback to defaults.

        If a property is not found in `props`, default values are provided for:
        - backpropagation (False)
        - grad_reg (None)
        - actual_output (None)
        - min_error (sys.float_info.max)
        - batch_size (1)
        - epochs (1)
        - scaler (None)
        - decay_rate (0)
        - risk (True),
        - exp_van_correction (False)
        - ewc_lambda (0.01)
        - 'l2_lambda (0.00)
        - 'beta1 (0.9)
        - 'beta2 (0.999)
        - 'epsilon (1e-08)

        Parameters:
        name (str): The name of the attribute to access.

        Returns:
        value: The value of the attribute from `props` or the default value.
        """
        return self.props.get(name, {
            'backpropagation': False,
            'grad_reg': None,
            'actual_output': None,
            'min_error': sys.float_info.max,
            'batch_size': 1,
            'epochs': 1,
            'scaler': None,
            'decay_rate': 0,
            'risk': True,
            'exp_van_correction': False,
            'ewc_lambda': 0.0,
            'l2_lambda': 0.0,
            'beta1': 0.9,
            'beta2': 0.999,
            'epsilon': 1e-08,
        }.get(name))

    def clone(self):
        """
        Create a deep copy of the current instance.

        Returns:
        OptimizationArgs: A new instance with the same arguments.
        """
        return OptimizationArgs(
            learning_rate=self.learning_rate,
            max_iter=self.max_iter,
            loss_function=self.loss_function,
            inputs=self.inputs[:],
            desired_output=self.desired_output[:],
            **self.props.copy()
        )

    def compute_error(self):
        """
        Compute the error between the actual and desired output using the
        loss function.

        Returns:
        float: The error computed by the loss function.
        """
        if self.loss_function is None:
            raise ValueError("Loss function not provided.")
        if self.actual_output is None or self.desired_output is None:
            raise ValueError(
                "Both actual_output and desired_output must be provided.")

        return self.loss_function(self.actual_output, self.desired_output)

    def batches(self):
        """
        Generator that yields batches of inputs and desired outputs.

        Yields:
        tuple: A batch of inputs and the corresponding desired output.
        """
        if self.inputs is None or self.desired_output is None:
            raise ValueError(
                "Both inputs and desired_output must be provided for batching.")

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
        c_0 = merged_d_o[0]
        c_1 = c_0.T
        c_2 = c_1.tolist()
        c_3 = [[x] for x in c_2]
        return c_3

    @staticmethod
    def split_inputs(merged_in):
        c_0 = merged_in.T
        c_1 = c_0.tolist()
        c_2 = [[x] for x in c_1]
        return c_2

    def __str__(self):
        """
        String representation of the class, showing key optimization arguments.

        Returns:
        str: A string summary of the main parameters.
        """
        inputs_shape = _get_shape(self.inputs)
        desired_output_shape = _get_shape(self.desired_output)
        props_to_str = [f"{key}={value}, " for key, value in
                        self.props.items()]
        return (
            f"OptimizationArgs("
            f"learning_rate={self.learning_rate}, "
            f"max_iter={self.max_iter}, "
            f"loss_function={self.loss_function}, "
            f"inputs_shape={inputs_shape}, "
            f"desired_output_shape={desired_output_shape}, "
            f"{props_to_str}"
            f")"
        )
