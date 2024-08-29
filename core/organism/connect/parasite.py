from core.cell import Cell, Linker, State, Seq, OptimizationArgs, \
    ParasiteEpochedOptimizer, Optimizer, OCell
from typing import List

from core.organism.connect import ParasiticLinker
from core.organism.connect.utils import get_cell_node, connect


class Parasite(Cell):
    def __init__(self, cells: List[Cell], optimizer: Optimizer = None):
        self.seq: Seq = connect(cells)
        super().__init__(self.seq, self.seq.first.arity,
                         0, optimizer=optimizer)

    def optimize(self, args: OptimizationArgs):
        self.optimizer(self, args)

    def p_optimize(self, args: OptimizationArgs):
        # print(args)
        self(args.merged_inputs)  # Forward pass
        children = self.seq.children
        args_clone = args.clone()

        for child in reversed(children[1:]):
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
        child = children[0]
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
        organism = Parasite(children, optimizer=optimizer)
        organism.set_optimization_risk(True)
        return organism



