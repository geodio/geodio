import tensor_bindings
class Tensor:
    def __init__(self, data, shape=None):
        """
        High-level constructor for AnyTensor. Automatically infers types.
        """
        if isinstance(data, list) and all(isinstance(x, float) for x in data):
            self.tensor = tensor_bindings.AnyTensor(data, shape)
        elif isinstance(data, list) and all(isinstance(x, int) for x in data):
            self.tensor = tensor_bindings.AnyTensor(data, shape)
        elif isinstance(data, float):
            self.tensor = tensor_bindings.AnyTensor(data)
        elif isinstance(data, int):
            self.tensor = tensor_bindings.AnyTensor(data)
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

    def __repr__(self):
        return f"<Tensor shape={self.shape()} data={self.data()}>"
