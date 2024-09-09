import numpy as np

from core.cell.operands.collections.activation_function. \
    base_activation_function import ActivationFunction


def softmax(x):
    """
    Parameters

    x: input matrix of shape (m, d)
    where 'm' is the number of samples (in case of batch gradient descent of size m)
    and 'd' is the number of features
    """
    z = x - np.max(x, axis=-1, keepdims=True)
    numerator = np.exp(z)
    denominator = np.sum(numerator, axis=-1, keepdims=True)
    softmax_r = numerator / denominator
    return softmax_r


def d_softmax(x):
    """
    Parameters

    x: input matrix of shape (m, d)
    where 'm' is the number of samples (in case of batch gradient descent of size m)
    and 'd' is the number of features
    """
    if len(x.shape) == 1:
        x = np.array(x).reshape(1, -1)
    else:
        x = np.array(x)
    m, d = x.shape
    a = softmax(x)
    tensor1 = np.einsum('ij,ik->ijk', a, a)
    tensor2 = np.einsum('ij,jk->ijk', a, np.eye(d, d))
    return tensor2 - tensor1


class SoftmaxActivation(ActivationFunction):
    def __init__(self, children, optimizer=None):
        super().__init__(children, optimizer)

        self._derivative = d_softmax

    def actual_forward(self, x, meta_args=None):
        return softmax(x)

    def clone(self) -> "SoftmaxActivation":
        return SoftmaxActivation([child.clone() for child in self.children],
                                 optimizer=self.optimizer.clone())

    def to_python(self) -> str:
        return "softmax"
