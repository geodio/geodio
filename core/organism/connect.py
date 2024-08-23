from typing import List

from core.cell import Cell, Linker, State, Seq, OptimizationArgs, \
    ParasiteEpochedOptimizer, Optimizer, Wrapper, transpose_of
from core.organism.node import Node


def get_cell_node(activation_function, dim_in, dim_out):
    return Cell(Node(1, dim_in, dim_out,
                     activation_function.clone()), 1, 2)


class ParasiticLinker(Linker):
    def __init__(self, cell_parasite: Cell, host_state: State):
        super().__init__(cell_parasite, host_state)

    host_state: State = property(lambda self: self.g)
    parasite: Cell = property(lambda self: self.f)

    def __call__(self, args, meta_args=None):
        x = self.g(args)
        r = self.f([x])
        return r

    def derive_unchained_g(self, by_weight, index):
        chain = Linker(self.f.derive(0, False), self.g)
        return chain

def make_parasitic_root(cell_host: Cell,
                        cell_parasite: Cell) -> ParasiticLinker:
    host_state: State = cell_host.get_state_weight()
    connected = ParasiticLinker(cell_parasite, host_state)
    return connected


def connect(cells: List[Cell]) -> Seq:
    if not cells:
        raise ValueError("The cells list cannot be empty.")

    roots = [cells[0]]  # Start with the first cell

    # Iterate through pairs of consecutive cells
    for i in range(1, len(cells)):
        # Take the previous cell (host) and the current cell (parasite)
        cell_host = cells[i - 1]
        cell_parasite = cells[i]

        # Make the parasitic root and add it to the roots list
        parasitic_root = make_parasitic_root(cell_host, cell_parasite)
        roots.append(parasitic_root)

    # Return a new Cell with the sequence of roots
    return Seq(roots)


class Parasite(Cell):
    def __init__(self, cells: List[Cell], optimizer: Optimizer = None):
        self.seq: Seq = connect(cells)
        super().__init__(self.seq, self.seq.first.arity,
                         0, optimizer=optimizer)

    def optimize(self, args: OptimizationArgs):
        n = self(args.inputs)  # Forward pass
        children = self.seq.children
        args_clone = args.clone()

        for child in reversed(children[1:]):
            child.optimize(args_clone)
            assert isinstance(child, ParasiticLinker), \
                f"Child is not of type ParasiticLinker"
            args_clone = args_clone.clone()
            args_clone.desired_output = [child.host_state.get()]

    @staticmethod
    def create_parasitic_organism(dim_in, dim_hidden, hidden_count, dim_out,
                                  activation_function, spread_point=-1,
                                  optimizer=None):
        input_node = get_cell_node(activation_function, dim_in, dim_hidden)
        optimizer = optimizer or ParasiteEpochedOptimizer()

        children = [input_node]
        for i in range(1, hidden_count + 1):
            hidden = get_cell_node(activation_function, dim_hidden, dim_hidden)
            children.append(hidden)

        output_node = get_cell_node(activation_function, dim_hidden, dim_out)
        children.append(output_node)
        organism = Wrapper(Parasite(children), optimizer=optimizer)
        organism.set_optimization_risk(True)
        return organism
