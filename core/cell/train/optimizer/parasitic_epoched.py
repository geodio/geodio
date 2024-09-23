from core.cell.train.optimizer.epoched import EpochedOptimizer


class ParasiteEpochedOptimizer(EpochedOptimizer):
    def train(self, model, optimization_args):
        model.p_optimize(optimization_args)