from core.cell.operands.collections.builtins.builtinbase import \
    BuiltinBaseFunction
from core.cell.operands.collections.builtins.conditionals import Condition

from core.cell.operands.operand import GLOBAL_BUILTINS, Operand


class If(BuiltinBaseFunction):
    def __init__(self, condition: Condition, children):
        if len(children) != 2:
            raise ValueError(
                "If operation requires exactly two children (then and else "
                "branches)")
        super().__init__(children, "if", 2)
        self.condition = condition

    def __call__(self, args, meta_args=None):
        # Evaluate the condition
        if self.condition(args, meta_args):
            return self.children[0](args, meta_args)  # Then branch
        else:
            return self.children[1](args, meta_args)  # Else branch

    def derive(self, index, by_weights=True):
        """
        The derivative of an If statement could depend on which branch is
        taken.

        :param index: index of variable/weight with respect to which
        derivative is taken
        :param by_weights: boolean, decides if derivative with respect to a
        weight or to a variable
        :return: the derivative operand;
        """
        # The derivative of an If statement could depend on which branch is
        # taken
        return If(
            self.condition.clone(),
            [
                self.children[0].derive(index, by_weights),
                self.children[1].derive(index, by_weights)
            ]
        )

    def clone(self):
        return If(
            self.condition.clone(),
            [child.clone() for child in self.children]
        )

    def to_python(self) -> str:
        return (f"({self.condition.to_python()}) "
                f"? ({self.children[0].to_python()}) "
                f": ({self.children[1].to_python()})")


def if_cond(condition: Condition, operand1: Operand, operand2: Operand):
    return If(condition, [operand1, operand2])


GLOBAL_BUILTINS["if_cond"] = if_cond
