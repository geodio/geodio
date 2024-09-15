from core.cell.operands.collections.builtins import BuiltinBaseFunction


class Seq(BuiltinBaseFunction):
    def __init__(self, children):
        super().__init__(children, "prod", len(children))

    def __call__(self, args, meta_args=None):
        for child in self.children[:-1]:
            child(args, meta_args=meta_args)
        if not self.children:
            return None
        r = self.children[-1](args, meta_args=meta_args)
        return r

    def derive(self, index, by_weights=True):
        return self.children[-1].derive(index, by_weights)

    def clone(self) -> "Seq":
        return Seq([child.clone() for child in self.children])

    def to_python(self) -> str:
        str_children = list(map(str, self.children))
        r = "\n".join(str_children)
        return r

    first = property(lambda self: self.children[0])
    last = property(lambda self: self.children[-1])

    def __len__(self) -> int:
        return len(self.children)

    def __iter__(self):
        return iter(self.children)

    def __getitem__(self, index):
        return self.children[index]
