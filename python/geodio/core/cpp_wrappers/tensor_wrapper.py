from geodio import geodio_bindings
import numpy as np

EMPTY_TENSOR = geodio_bindings.AnyTensor()
tensor = type(EMPTY_TENSOR)

def infer_shape(data):
    if isinstance(data, list):
        return [len(data)] + infer_shape(data[0]) if data else []
    else:
        return []

class Tensor:
    def __init__(self, data, shape=None):
        """
        High-level constructor for AnyTensor. Automatically infers types.
        """
        if isinstance(data, np.ndarray):
            # Convert numpy array to list and infer shape
            if len(data.shape) == 1:
                shape = [data.shape[0]] if shape is None else shape
            else:
                shape = list(data.shape) if shape is None else shape
            data = data.flatten().tolist()
        if shape is None and isinstance(data, list):
            shape = infer_shape(data)
        if isinstance(data, list) and all(isinstance(x, float) for x in data):
            self.tensor = geodio_bindings.AnyTensor(data, shape)
        elif isinstance(data, list) and all(isinstance(x, int) for x in data):
            self.tensor = geodio_bindings.AnyTensor(data, shape)
        elif isinstance(data, float):
            self.tensor = geodio_bindings.AnyTensor(data)
        elif isinstance(data, int):
            self.tensor = geodio_bindings.AnyTensor(data)
        elif isinstance(data, tensor):
            self.tensor = geodio_bindings.AnyTensor(data)
        else:
            raise ValueError("Unsupported type for Tensor")

    def __add__(self, other):
        """
        Add two tensors, providing an easy-to-use interface.
        """
        if not isinstance(other, Tensor):
            raise ValueError("The operand must be a Tensor object")
        return Tensor(self.tensor + other.tensor)

    def __sub__(self, other):
        return Tensor(self.tensor - other.tensor)

    def __mul__(self, other):
        return Tensor(self.tensor * other.tensor)

    def __truediv__(self, other):
        return Tensor(self.tensor / other.tensor)

    def matmul(self, other):
        """
        Perform matrix multiplication.
        """
        return Tensor(self.tensor.matmul(other.tensor))

    def sum(self, axis=None):
        """
        Compute the sum across the specified axis.
        """
        if axis is None:
            axis = [0]
        return Tensor(self.tensor.sum(axis))

    def transpose(self, axis=None):
        """
        Transpose the tensor along the specified axes.
        """
        if axis is None:
            axis = [0]
        return Tensor(self.tensor.transpose(axis))

    def shape(self):
        """
        Return the shape of the tensor.
        """
        return self.tensor.shape()

    def data(self):
        """
        Return the data of the tensor in its native Python type.
        """
        if self.tensor.is_float():
            return self.tensor.get_float()
        elif self.tensor.is_int():
            return self.tensor.get_int()
        else:
            raise TypeError("Unsupported tensor type")

    def __str__(self):
        return f"{str(self.tensor)}"

    def __repr__(self):
        return f"<Tensor {self.tensor}>"
