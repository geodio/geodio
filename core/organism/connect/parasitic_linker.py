import numpy as np

from core.cell import Linker, Cell, State, OptimizationArgs
from core.cell.locker import Locker


class ParasiticLinker(Linker):
    def __init__(self, cell_parasite: Cell, host_state: State):
        super().__init__(cell_parasite, host_state)
        self._pl_locked_parasite = None

    host_state: State = property(lambda self: self.g)
    parasite: Cell = property(lambda self: self.f)

    def __call__(self, args, meta_args=None):
        x = self.g(args)
        r = self.f([x])
        return r

    def get_sub_operands(self):
        return [self.f]

    def derive_uncached(self, index, by_weight=True):
        """
        (f(g(x)))' = f'(g(x)) * g'(x)
        :param index:
        :param by_weight:
        :return:
        """
        non_parasitic = self.f.link(self.host_state.cell)
        derivative = non_parasitic.derive_uncached(index, by_weight=by_weight)
        return derivative

    def _get_linker_locked_parasite(self):
        if self._pl_locked_parasite is None:
            self._pl_locked_parasite = ParasiticLinker(Locker(self.parasite,
                                                              self.arity),
                                                       self.host_state)

    linker_locked_parasite: "ParasiticLinker" = (
        property(lambda self:
                 self._get_linker_locked_parasite())
    )

    def optimize(self, args: OptimizationArgs):
        # Optimize parasite
        self.parasite.optimize(args)

        # Optimize host state
        # self.optimize_host_state(args)

    def optimize_host_state(self, args: OptimizationArgs):
        """
        Here we make a custom MSE multivariate gradient function.
        It is totally different form the usual MSE gradient function.
        And we also update the state with gradient descent.
        I do not understand entirely why it works, but it does.

        :param args: optimization arguments
        :return: None
        """
        initial_host_state = self.host_state.get().copy()
        jacobian_results = self.parasite.d(0)([initial_host_state])
        predicted = self(args.inputs)
        # TODO DOES NOT WORK ANYMORE
        # print('jacobian_results', np.shape(jacobian_results))
        # print('initial_host_state', np.shape(initial_host_state))
        # print('desired_output', np.shape(args.desired_output))
        # print('predicted', np.shape(predicted))

        diff = (
                np.array(args.desired_output)[:, 0, :].T -
                np.array(predicted)
        )
        # diff = diff[:, :, np.newaxis]
        per_instance_grad = diff * jacobian_results
        per_instance_grad = np.mean(per_instance_grad, axis=0)
        k = initial_host_state.shape[0]
        per_instance_grad = np.tile(per_instance_grad, (k, 1))
        mg = per_instance_grad
        niu = args.learning_rate / args.batch_size
        new_state = initial_host_state + niu * mg
        self.host_state.set(new_state)

        # TODO VERIFY IF OPTIMIZATION OF STATE IS WORKING
        # CHECK IF THERE IS LOSS DESCENT
