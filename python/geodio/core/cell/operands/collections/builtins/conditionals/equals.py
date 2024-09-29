from geodio.core.cell.operands.collections.builtins.conditionals.condition import \
    Condition
from geodio.core.cell.operands.operand import GLOBAL_BUILTINS, Operand


class Equals(Condition):
    def __init__(self, children):
        if len(children) != 2:
            raise ValueError("Equals operation requires exactly two operands")
        super().__init__(children)

    def __call__(self, args, meta_args=None):
        return self.children[0](args, meta_args) == self.children[1](args,
                                                                     meta_args)

    def to_python(self) -> str:
        return f"({self.children[0].to_python()}) == ({self.children[1].to_python()})"

    def get_operand_type(self):
        return geodio.geodio_bindings.OperandType.Equals


def equals(operand1: Operand, operand2: Operand):
    return Equals([operand1, operand2])


GLOBAL_BUILTINS["equals"] = equals
