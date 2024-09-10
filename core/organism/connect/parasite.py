import numpy as np

from core.cell import Cell, Seq, OptimizationArgs, \
    ParasiteEpochedOptimizer, Optimizer, Operand, MSE
from core.cell import LinearActivation
from core.organism.connect import ParasiticLinker
from core.organism.connect.utils import get_cell_node, connect


class Parasite(Cell):
    def __init__(self, seq: Seq, optimizer: Optimizer = None):
        self.seq: Seq = seq
        super().__init__(self.seq, self.seq.first.arity,
                         0, optimizer=optimizer)

    def optimize(self, args: OptimizationArgs):
        self.optimizer(self, args)

    def p_optimize(self, args: OptimizationArgs):
        # print(args)
        self(args.merged_inputs)  # Forward pass
        args_clone = args.clone()
        n = len(self.seq)
        for i in range(n - 1, 0, -1):
            child = self.seq[i]
            # Verify child
            assert isinstance(child, ParasiticLinker), \
                f"Child is not of type ParasiticLinker"
            # Prepare inputs for this child
            inputs = args.split_inputs(child.host_state.get())
            args_clone.inputs = inputs

            # Optimize this child
            child.optimize(args_clone)

            # Prepare outputs for previous child
            args_clone = args_clone.clone()
            merged_state = [child.host_state.get()]
            split_state = args.split_desired_output(merged_state)
            args_clone.desired_output = split_state

        # Prepare inputs for input node
        child = self.seq[0]
        inputs = args.inputs
        args_clone.inputs = inputs

        # Optimize input node
        child.optimize(args_clone)

        # Calculate error
        self.compute_error(args)

    def compute_error(self, args):
        y_pred = [self(x_inst) for x_inst in args.inputs]
        if args.desired_output is None:
            pass
        else:
            self.error = args.loss_function(args.desired_output, y_pred)
        return self.error

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
        seq: Seq = connect(children)
        organism = Parasite(seq, optimizer=optimizer)
        organism.set_optimization_risk(True)
        return organism

    @staticmethod
    def create_recursive_parasitic_organism(root: Operand,
                                            optimizer=None):
        return NotImplemented


def main():
    ns = np.arange(start=1, stop=81, step=3)
    args = [np.arange(n) for n in ns]
    desired = [[n * (n + 1) / 2] for n in ns]

    root = Node(1, 2, 1, LinearActivation(), scalar_output=True)
    parasite = Parasite.create_recursive_parasitic_organism(root)
    opt_args = OptimizationArgs(
        max_iter=1,
        loss_function=MSE(),
        inputs=args,
        desired_output=desired,
        batch_size=5,
        epochs=100
    )

    for arg in args:
        print(parasite(arg))
    parasite.optimize(opt_args)
    for arg in args:
        print(parasite(arg))


if __name__ == '__main__':
    main()
