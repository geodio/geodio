from core.cell.operands.collections.builtins.seq import Seq
from core.cell.operands.operand import Operand
from core.cell.operands.variable import Variable


class RecSeq(Seq):
    def __init__(self, operand: Operand):
        super().__init__([operand])
        self._last_args = None
        self._entry_point = Variable(0)
        self._sliding_operand = self._entry_point.link(operand)

    def __call__(self, args, meta_args=None):
        self._last_args = args
        n = len(self._last_args)
        for i in range(n - 1):
            self._entry_point.value = i
            self._sliding_operand(args, meta_args)
        self._entry_point.value = n - 1
        r = self._sliding_operand(args, meta_args)
        return r

    def __len__(self) -> int:
        return len(self._last_args)

    def __iter__(self):
        n = len(self._last_args)
        for i in range(n):
            self._entry_point.value = i
            yield self._sliding_operand

    def __getitem__(self, index):
        if isinstance(index, slice):
            return NotImplemented
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            self._entry_point.value = int(index)
            return self._sliding_operand
        raise IndexError(f"Index out of range for index {index}")

    def derive(self, index, by_weights=True):
        return NotImplemented

    def clone(self) -> "Seq":
        return NotImplemented

    first = property(lambda self: self[0])
    last = property(lambda self: self[-1])

