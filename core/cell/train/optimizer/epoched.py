from core import logger
from core.cell.train.optimizer.default_optimizer import Optimizer


class EpochedOptimizer(Optimizer):
    def __init__(self, optimization=None):
        self.optimizer = None
        super().__init__(optimization)

    def train(self, model, optimization_args):
        if self.optimizer is None:
            self.optimizer = self.make_optimizer(model, optimization_args)
        else:
            self.optimizer.update_optim_args(optimization_args)
        self.optimizer.optimize()

    def __call__(self, model, optimization_args):
        a = optimization_args
        extra = a.extra_action
        if optimization_args.backpropagation:
            self.optimization = "backpropagation"
        niu = a.learning_rate
        decay_rate = a.decay_rate
        logger.logging.debug("Organism Optimization Started.")
        for epoch in range(a.epochs):
            epoch_loss = 0
            its_debug_time = epoch % 25 == 0
            if its_debug_time:
                logger.logging.debug(f"Epoch {epoch}")
            for X_batch, y_batch in a.batches():
                input_data = X_batch
                desired_output = y_batch

                optimization_args = a.clone()
                optimization_args.learning_rate = niu
                optimization_args.inputs = input_data
                optimization_args.desired_output = desired_output
                self.train(model, optimization_args)
                epoch_loss += model.error
            epoch_loss /= a.batch_size
            if its_debug_time:
                logger.logging.debug(f"LOSS {model.error}")
                if extra:
                    extra()
            niu -= niu * decay_rate
