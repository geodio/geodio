from geodio.core.cell.operands.variable import MetaVariable, MetaAssignment

import uuid

from geodio.core.cell.operands.operand import Operand
from geodio.core.utils import flatten


def verify_equal_children(operand_a: Operand, operand_b: Operand) -> bool:
    children_a = operand_a.get_sub_operands()
    children_b = operand_b.get_sub_operands()
    if len(children_a) != len(children_b):
        return False
    for a, b in zip(children_a, children_b):
        if a != b:
            return False
    return True


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
                    meta_assignment = MetaAssignment(meta_var_id, reduced_child)
                    meta_assignments.append(meta_assignment)

        return meta_variables, meta_assignments

    # Recursively reduce the children first
    children = operand.get_sub_operands()
    meta_variables, meta_assignments = collect_equal_children(children)

    # Create new children with MetaVariables
    new_children = [meta_variables.get(child, reduce(child)) for child in
                    children]

    # Recreate the operand with the new children
    reduced_operand = operand.__class__(*new_children)

    # Attach meta assignments to the reduced operand
    # If not, this part needs to be adjusted according to actual implementation
    reduced_operand.meta_assignments = meta_assignments

    return reduced_operand


def get_predicted(X, cell):
    return flatten([cell(x_inst) for x_inst in X])
