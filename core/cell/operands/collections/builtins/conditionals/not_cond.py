from core.cell.operands.collections.builtins.conditionals.condition import \
    Condition
from core.cell.operands.operand import GLOBAL_BUILTINS, Operand


class Not(Condition):
    def __init__(self, child):
        if len(child) != 1:
            raise ValueError("Not operation requires exactly one operand")
        super().__init__(child)

    def __call__(self, args, meta_args=None):
        return not self.children[0](args, meta_args)

    def to_python(self) -> str:
        return f"not ({self.children[0].to_python()})"


def not_cond(operand: Operand):
    return Not([operand])


GLOBAL_BUILTINS["not"] = not_cond
