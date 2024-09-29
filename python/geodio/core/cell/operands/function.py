from geodio.core.cell.operands.operand import Operand
from geodio.core.cell.operands.utility import verify_equal_children


class Function(Operand):
    def __init__(self, arity, value, children=None, is_childless=False):
        super().__init__(arity)
        self.value = value
        if is_childless:
            self.children = None
        else:
            self.children = children if children is not None else []
        self.is_childless = is_childless

    def __call__(self, args, meta_args=None):
        if not self.is_childless:
            func_args = [child(args, meta_args) for child in self.children]
            return self.value(*func_args)
        return self.value(*args)

    def __invert__(self):
        return None  # Replace with actual inversion logic

    def clone(self) -> "Function":
        return Function(self.arity, self.value, [child.clone() for child in
                                                 self.children])

    def to_python(self) -> str:
        args = [child.to_python() for child in self.children]
        return f"{self.value.__name__}({', '.join(args)})"

    def get_weights_local(self):
        weights = []
        for child in self.children:
            weights.extend(child.get_weights_local())
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

    def __eq__(self, other):
        if isinstance(other, Function):
            return (
                    self.value.__name__ == other.value.__name__ and
                    verify_equal_children(self, other)
                    )
        return False


class PassThrough(Operand):
    def __invert__(self):
        pass

    def __init__(self, arity):
        super().__init__(arity)

    def __call__(self, args, meta_args=None):
        return args

    def derive(self, index, by_weight=True):
        return self

    def set_weights(self, new_weights):
        pass

    def clone(self):
        return PassThrough(self.arity)

    def to_python(self) -> str:
        return "X"

    def __eq__(self, other):
        return isinstance(other, PassThrough)


class Collector(Operand):
    def __init__(self, arity, children):
        super().__init__(arity)
        self.children = children

    def __call__(self, args, meta_args=None):
        r = [child(args, meta_args) for child in self.children]
        return r

    def derive(self, index, by_weight=True):
        return Collector(self.arity, [child.derive(index, by_weight) for child
                         in (
                self.children)])

    def set_weights(self, new_weights):
        raise NotImplementedError

    def clone(self):
        return Collector(self.arity, [child.clone() for child in self.children])

    def __invert__(self):
        raise NotImplementedError

    def to_python(self) -> str:
        return "[" + ', '.join(self.children) + "]"

    def __eq__(self, other):
        raise NotImplementedError
