from typing import Optional, Dict, Any, List, Tuple

from geodio.core.cell.operands.collections.builtins import BuiltinBaseFunction
from geodio.core.cell.operands.operand import Operand


def expand_labels(children: List[Operand]) -> Tuple[List[Operand], Dict[str, int]]:
    """
    Expands labels by inserting their children and updates the label positions.
    """
    expanded_children = []
    label_positions: Dict[str, int] = {}
    for idx, child in enumerate(children):
        if isinstance(child, Label):
            # Store the label position in the dictionary
            label_positions[child.str_id] = len(expanded_children)
            # Add the label itself to the expanded list
            expanded_children.append(child)
            # Add the children of the label to the expanded list
            expanded_children.extend(child.children)
        else:
            expanded_children.append(child)
    return expanded_children, label_positions

class BoxInt:
    def __init__(self, initial_value=0):
        self.__i = initial_value

    @property
    def i(self):
        return self.__i

    @i.setter
    def i(self, value):
        self.__i = value

    def inc(self):
        self.__i += 1

class Seq(BuiltinBaseFunction):
    def __init__(self, children):
        super().__init__(children, "seq", len(children))

    def __call__(self, args, meta_args=None):
        expanded_children, label_positions = expand_labels(self.children)
        box_int = BoxInt()
        result = 0
        while box_int.i < len(expanded_children):
            child = expanded_children[box_int.i]
            if isinstance(child, Label):
                # Skip labels since their children are already in the
                # expanded list
                box_int.inc()
                continue
            elif isinstance(child, Jump):
                # Get the label ID to jump to and update the index to that
                # position
                label_id = child.str_id
                if label_id in label_positions:
                    box_int.i = label_positions[label_id]
                else:
                    # Straight up return the jump statement
                    # Since maybe a parent SEQ may know how to deal with it.
                    return child
            else:
                # Call the operand and continue to the next one
                result = child(args, meta_args=meta_args)
                if isinstance(result, Jump):
                    label_id = result.str_id
                    if label_id in label_positions:
                        box_int.i = label_positions[label_id]
                    else:
                        raise RuntimeError(
                            f"Jump to undefined label: {label_id}")
                else:
                    box_int.inc()

        return result

    # def __call__(self, args, meta_args=None):
    #     for child in self.children[:-1]:
    #         child(args, meta_args=meta_args)
    #     if not self.children:
    #         return None
    #     r = self.children[-1](args, meta_args=meta_args)
    #     return r

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

    def get_operand_type(self):
        return geodio.geodio_bindings.OperandType.Seq


class Label(Operand):
    def __init__(self, str_id, children):
        super().__init__(0)
        self.str_id = str_id
        self.children = children

    def derive(self, index, by_weights=True):
        pass

    def __call__(self, args, meta_args: Optional[Dict[str, Any]] = None):
        return self.str_id

    def __invert__(self):
        pass

    def clone(self) -> "Label":
        return Label(self.str_id, [kid.clone() for kid in self.children])

    def to_python(self) -> str:
        pass

    def get_operand_type(self):
        return geodio.geodio_bindings.OperandType.Label


class Jump(Operand):
    def __init__(self, str_id):
        super().__init__(0)
        self.str_id = str_id

    def derive(self, index, by_weights=True):
        pass

    def __call__(self, args, meta_args: Optional[Dict[str, Any]] = None):
        return self

    def __invert__(self):
        pass

    def clone(self) -> "Jump":
        return Jump(self.str_id)

    def to_python(self) -> str:
        pass

    def get_operand_type(self):
        return geodio.geodio_bindings.OperandType.Jump
