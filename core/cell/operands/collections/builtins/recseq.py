from core.cell.operands.collections.builtins.linker import Linker
from core.cell.operands.stateful import Stateful
from core.cell.operands.collections.builtins.seq import Seq
from core.cell.operands.operand import Operand
from core.cell.operands.variable import Variable


class RecSeq(Seq):
    def __init__(self, operand: Operand):
        super().__init__([operand])
        self._last_args = None
        self._entry_point = Variable(0)

        self._is_stateful = isinstance(operand, Stateful)
        if isinstance(operand, Stateful):
            # self._operand is not needed otherwise.
            self._operand: Stateful = operand

        self._sliding_operand: Linker = self._entry_point.link(operand)

    def __call__(self, args, meta_args=None):
        """
        If the operand that is recursively sequenced is Stateful,
        all historical records of states are cleared, and, for every argument
        in args, a new checkpoint is created.
        :param args: A list of arguments to evaluate the RecSeq operand with.
        :param meta_args: Additional metadata arguments, if any.
        :return: The last result obtained after evaluating the operand with
        all the provided arguments.
        """

        if self._is_stateful:
            self._operand.clear_checkpoints(0)
        self._last_args = args
        n = len(self._last_args)
        for i in range(n - 1):
            self._entry_point.value = i
            self._sliding_operand(args, meta_args)
            if self._is_stateful:
                self._operand.mark_checkpoint()
        self._entry_point.value = n - 1
        r = self._sliding_operand(args, meta_args)
        if self._is_stateful:
            self._operand.mark_checkpoint()
        return r

    def __len__(self) -> int:
        return len(self._last_args)

    def __iter__(self):
        n = len(self._last_args)
        for i in range(n):
            self._entry_point.value = i
            if self._is_stateful:
                self._operand.use_checkpoint(i)
            yield self._sliding_operand.f

    def __getitem__(self, index):
        if isinstance(index, slice):
            return NotImplemented
        if isinstance(index, int):
            if index < 0:
                index += len(self)
            self._entry_point.value = int(index)
            if self._is_stateful:
                self._operand.use_checkpoint(index)
            return self._sliding_operand.f
        raise IndexError(f"Index out of range for index {index}")

    def derive(self, index, by_weights=True):
        return NotImplemented

    def clone(self) -> "Seq":
        return NotImplemented

    first = property(lambda self: self[0])
    last = property(lambda self: self[-1])

