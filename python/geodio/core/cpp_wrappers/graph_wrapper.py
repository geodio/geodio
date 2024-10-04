from geodio import geodio_bindings
from geodio.core.cpp_wrappers.tensor_wrapper import Tensor
from geodio.geodio_bindings import Operand

class GraphWrapper:
    def __init__(self):
        """
        Initialize a GraphWrapper that contains a C++ ComputationalGraph
        and keeps track of the max ID for operands.
        """
        self.graph = geodio_bindings.ComputationalGraph()
        self.max_id = -1  # so the root gets always id 0

    def add_operand(self, operand_type, graph_id, children):
        """
        Add an operand and its children to the graph.

        Args:
            operand_type (OperandType):
            graph_id (int):
            children (List[int]):

        Returns:
            None
        """
        operand = Operand(operand_type, graph_id, children)
        self.graph.add_operand(graph_id, operand)

    def add_constant(self, graph_id, tensor: Tensor):
        self.graph.add_constant(graph_id, tensor.tensor)

    def add_weight(self, graph_id, tensor: Tensor):
        self.graph.add_weight(graph_id, tensor.tensor)

    def add_var_map(self, graph_id, value: int):
        self.graph.add_var_map(graph_id, value)

    def next_id(self):
        """
        Generate the next unique ID for operands in the graph.

        Returns:
            int: The next unique operand ID.
        """
        self.max_id += 1
        return self.max_id
