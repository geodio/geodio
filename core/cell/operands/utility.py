from typing import Optional, Dict, Any

from core.cell.operands.variable import MetaVariable

import uuid

from core.cell.operands.operand import Operand


def verify_equal_children(operand_a: Operand, operand_b: Operand) -> bool:
    children_a = operand_a.get_children()
    children_b = operand_b.get_children()
    if len(children_a) != len(children_b):
        return False
    for a, b in zip(children_a, children_b):
        if a != b:
            return False
    return True


class MetaAssignment(Operand):
    def __init__(self, meta_variable: str,
                 original_operand: Operand, attache: Operand):
        super().__init__(attache.arity)
        self.meta_variable = meta_variable
        self.original_operand = original_operand
        self.attache = attache

    def __call__(self, args, meta_args: Optional[Dict[str, Any]] = None):
        meta_args[self.meta_variable] = self.original_operand(args, meta_args)
        return self.attache(args, meta_args)

    def __invert__(self):
        pass

    def clone(self) -> "Operand":
        pass

    def to_python(self) -> str:
        return f"""
        M{self.meta_variable} = {self.original_operand.to_python()}
        return {self.attache.to_python()}
        """

    def derive(self, index, by_weights=True):
        # TODO
        pass

    def get_children(self):
        return [self.original_operand, self.attache]

    def __eq__(self, other):
        if isinstance(other, MetaAssignment):
            return (self.meta_variable == other.meta_variable
                    and self.original_operand == other.original_operand
                    and self.attache == other.attache)
        return False


def reduce(operand: Operand) -> Operand:
    # TODO
    def unique_id():
        return str(uuid.uuid4())

    def collect_equal_children(children):
        meta_variables = {}
        meta_assignments = []

        for i, child in enumerate(children):
            if child not in meta_variables:
                # Recursively reduce the child
                reduced_child = reduce(child)
                equal_children = [j for j in range(i, len(children)) if
                                  reduce(children[j]) == reduced_child]

                if len(equal_children) > 1:
                    meta_var_id = unique_id()
                    meta_var = MetaVariable(meta_var_id)
                    for idx in equal_children:
                        meta_variables[children[idx]] = meta_var
                    meta_assignment = MetaAssignment(meta_var, reduced_child)
                    meta_assignments.append(meta_assignment)

        return meta_variables, meta_assignments

    # Recursively reduce the children first
    children = operand.get_children()
    meta_variables, meta_assignments = collect_equal_children(children)

    # Create new children with MetaVariables
    new_children = [meta_variables.get(child, reduce(child)) for child in
                    children]

    # Recreate the operand with the new children
    reduced_operand = operand.__class__(*new_children)

    # Attach meta assignments to the reduced operand
    # Assuming there is a method or a way to attach meta assignments to an operand
    # If not, this part needs to be adjusted according to actual implementation
    reduced_operand.meta_assignments = meta_assignments

    return reduced_operand
