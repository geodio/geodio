import tensorflow as tf

from src.cell.operands.operand import Operand


class Weight(Operand):
    def __init__(self, weight=0.0):
        super().__init__(0)
        self.weight = tf.Variable([weight], dtype=tf.float32)

    def __call__(self, args):
        return self.weight[0].numpy()

    def d(self, var_index):
        return ZERO_WEIGHT

    def __invert__(self):
        return None

    def get_weights(self):
        return self.weight

    def set_weights(self, new_weights):
        self.weight.assign(new_weights[0])

    def clone(self):
        return Weight(self.weight[0])

    def to_python(self) -> str:
        return str(self.weight)


ZERO_WEIGHT = Weight(0)
