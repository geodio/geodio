from core.cell import *
from core.organism.node import Node
from core.organism.activation_function import SigmoidActivation


def main():
    activation = SigmoidActivation()
    node_a = Node(1, 1, 1, activation)
    node_b = Node(1, 1, 1, activation)
    cell_a = Cell(node_a, 1, 1)
    model = cell_a.get_state_weight()
    print(model([np.array([7])]))
    model = model.link(node_b)
    cell_a.update(np.array([800]))
    print("STATE:", cell_a.state)
    print(model([np.array([7])]))
    model = model.link(cell_a)
    print("STATE:", cell_a.state)
    print(model([np.array([7])]))
    print("STATE:", cell_a.state)
    print(model([np.array([7])]))

    print("STATE:", cell_a.state)
    print(model([np.array([7])]))


if __name__ == '__main__':
    main()
