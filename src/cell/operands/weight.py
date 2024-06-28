from src.cell.operands.operand import Operand


class Weight(Operand):
    def __init__(self, weight=0):
        super().__init__(0)
        self.weight = weight

    def __call__(self, args):
        return self.weight

    def d(self, var_index):
        return ZERO_WEIGHT

    def __invert__(self):
        return None

    def clone(self) -> "Weight":
        return Weight(self.weight)

    def to_python(self) -> str:
        return str(self.weight)

ZERO_WEIGHT = Weight(0)