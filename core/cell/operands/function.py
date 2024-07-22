from typing import Optional

from core.cell.operands.operand import Operand


class Function(Operand):
    def __init__(self, arity, value, children=None):
        super().__init__(arity)
        self.value = value
        self.children = children if children is not None else []

    def __call__(self, args):
        func_args = [child(args) for child in self.children]
        return self.value(*func_args)

    def d(self, var_index) -> Optional[Operand]:
        return None

    def __invert__(self):
        return None  # Replace with actual inversion logic

    def clone(self) -> "Function":
        return Function(self.arity, self.value, [child.clone() for child in
                                                 self.children])

    def to_python(self) -> str:
        args = [child.to_python() for child in self.children]
        return f"{self.value.__name__}({', '.join(args)})"

    def get_weights(self):
        weights = []
        for child in self.children:
            weights.extend(child.get_weights())

        for i, weight in enumerate(weights):
            weight.w_index = i

        return weights

    def set_weights(self, new_weights):
        offset = 0
        for child in self.children:
            child_weights = child.get_weights()
            num_weights = len(child_weights)
            if num_weights > 0:
                child.set_weights(new_weights[offset:offset + num_weights])
                offset += num_weights

    def derive(self, index, by_weight=True):
        return None


class PassThrough(Operand):
    def __invert__(self):
        pass

    def __init__(self, arity):
        super().__init__(arity)

    def __call__(self, args):
        return args

    def derive(self, index, by_weight=True):
        return self

    def set_weights(self, new_weights):
        pass

    def clone(self):
        return PassThrough(self.arity)

    def to_python(self) -> str:
        return "<->"