from core.cell.cell import Operand, Cell


class Locker(Cell):
    def __init__(self, root: Operand, arity):
        super().__init__(root, arity, 1)

    def __call__(self, args, meta_args=None):
        return self.root(args)

    def derive_uncached(self, index, by_weight=True):
        derivative = self.root.derive(index, by_weight)
        Locker(derivative, self.arity)
        return derivative

    def set_weights(self, new_weights):
        pass

    def get_weights_local(self):
        return []

    def clone(self):
        return Locker(self.root.clone(), self.arity)

    def __invert__(self):
        pass

    def to_python(self) -> str:
        return "LOCKED[]"

    def __eq__(self, other):
        return (isinstance(other, Locker)
                and self.root == other.root
                and self.arity == other.arity)