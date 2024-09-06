from typing import Optional, Dict, Any, Iterable, TypeVar, Protocol

from core.cell import Operand, OptimizableOperand, Optimizer, Add

T = TypeVar('T', bound=Operand)


class SizedIterable(Protocol[T]):
    def __len__(self) -> int:
        ...

    def __iter__(self) -> Iterable[T]:
        ...


class Aggregation(OptimizableOperand):
    def __init__(self, operand: Operand, optimizer: Optimizer = None):
        if operand is None:
            raise ValueError("Operand cannot be None")
        arity = operand.arity
        super().__init__(arity, optimizer)
        self.__aggregation = arity
        self.__operand = operand
        self.__aggregates = None

    def set_aggregates(self, aggregates: SizedIterable):
        if len(aggregates) != self.__aggregation:
            raise ValueError("Number of Aggregates does not fit operand arity")
        self.arity = max(list(map(lambda x: x.arity, self.__aggregates)))
        self.__aggregates = aggregates

    def delete_aggregates(self):
        self.__aggregates = None
        self.arity = 0

    def __call__(self, args, meta_args: Optional[Dict[str, Any]] = None):
        if self.__aggregates is None:
            raise ValueError("Aggregates have not been initialized")
        if len(self.__aggregates) != self.__aggregation:
            raise ValueError("Number of Aggregates does not fit operand arity")
        _op_args = [aggr[args, meta_args] for aggr in self.__aggregates]
        r = self.__operand(_op_args, meta_args)
        return r

    def derive_uncached(self, index, by_weights):
        terms = []
        for i in range(self.__aggregation):
            clone: Aggregation = self.clone()
            # TODO PROPERLY IMPLEMENT THIS
            clone.__aggregates[i] = self.__aggregates[i].derive(index,
                                                                by_weights)
            clone.__operand = self.__operand.d(i)
            terms.append(clone)
        derivative = Add(terms, self.arity)
        return NotImplemented

    def get_sub_operands(self):
        if self.__aggregates is None:
            raise ValueError("Aggregates have not been initialized")
        kids = [aggr for aggr in self.__aggregates]
        kids.append(self.__operand)
        return kids

    def clone(self) -> "Aggregation":
        cloned = Aggregation(self.__operand.clone(), self.optimizer.clone())
        cloned.__aggregates = [aggr.clone() for aggr in self.__aggregates]
        return cloned

    def to_python(self) -> str:
        return NotImplemented
