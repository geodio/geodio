from geodio.core.cell.operands.collections.builtins.conditionals.condition import \
    Condition

from geodio.core.cell.operands.operand import GLOBAL_BUILTINS, Operand


class And(Condition):
    def __init__(self, children):
        super().__init__(children)

    def __call__(self, args, meta_args=None):
        return all(child(args, meta_args) for child in self.children)

    def to_python(self) -> str:
        return " and ".join([
            f"({child.to_python()})" for child in self.children])


def and_cond(operand1: Operand, operand2: Operand):
    return And([operand1, operand2])


GLOBAL_BUILTINS["and"] = and_cond
