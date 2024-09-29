from geodio.core.cell.operands.collections.builtins.conditionals.condition import \
    Condition
from geodio.core.cell.operands.operand import GLOBAL_BUILTINS, Operand


class SmallerThanOrEqual(Condition):
    def __init__(self, children):
        if len(children) != 2:
            raise ValueError("SmallerThanOrEqual operation requires exactly "
                             "two operands")
        super().__init__(children)

    def __call__(self, args, meta_args=None):
        return self.children[0](args, meta_args) <= self.children[1](args,
                                                                     meta_args)

    def to_python(self) -> str:
        return (f"({self.children[0].to_python()}) "
                f"<= ({self.children[1].to_python()})")

    def get_operand_type(self):
        return geodio.geodio_bindings.OperandType.LessThanOrEqual


def smaller_than_or_equals(operand1: Operand, operand2: Operand):
    return SmallerThanOrEqual([operand1, operand2])


GLOBAL_BUILTINS["smaller_than_or_equals"] = smaller_than_or_equals
