from typing import List

from geodio.core.cell import OCell, Cell, Seq, State
from geodio.core.organism.connect import ParasiticLinker


def get_cell_node(activation_function, dim_in, dim_out):
    return OCell(Node(1, dim_in, dim_out,
                      activation_function.clone()), 1, 2)


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


def make_parasitic_root(cell_host: Cell,
                        cell_parasite: Cell) -> ParasiticLinker:
    host_state: State = cell_host.get_state_weight()
    connected = ParasiticLinker(cell_parasite, host_state)
    return connected
