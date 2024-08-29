from core.cell.operands.collections.builtins.conditionals.condition import \
    Condition
from core.cell.operands.operand import GLOBAL_BUILTINS, Operand


class Or(Condition):
    def __init__(self, children):
        super().__init__(children)

    def __call__(self, args, meta_args=None):
        return any(child(args, meta_args) for child in self.children)

    def to_python(self) -> str:
        return " or ".join(
            [f"({child.to_python()})" for child in self.children])


def or_cond(operand1: Operand, operand2: Operand):
    return Or([operand1, operand2])


GLOBAL_BUILTINS["or"] = or_cond
