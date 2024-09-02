from core import logger
from core.cell.train.optimizer.default_optimizer import Optimizer
from core.cell.train.optimizer.optimization.default_optimization import \
    Optimization


class EpochedOptimizer(Optimizer):
    def __init__(self, optimization=None):
        super().__init__(optimization)

    def train(self, model, optimization_args):
        optimizer = self.make_optimizer(model, optimization_args)
        optimizer.optimize()

    def __call__(self, model, optimization_args):
        a = optimization_args
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
            niu -= niu * decay_rate
