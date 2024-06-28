from src.cell.operands.weight import Weight


class Constant(Weight):
    def __init__(self, weight):
        super().__init__(weight)
        self.__value = weight

    def __call__(self, x):
        return self.__value

    def d(self, dx):
        return ZERO

    @staticmethod
    def from_weight(w: Weight):
        return Constant(w.weight)


ZERO = Constant(0)
ONE = Constant(1)
MINUS_ONE = Constant(-1)
