import numpy as np

from core.cell.train.optimizer.optimization.default_optimization \
    import Optimization


class BackpropagationOptimization(Optimization):

    def calculate_gradients(self):
        X_batch = np.array([x[0] for x in self.input])
        y_batch = np.array([y[0] for y in self.desired_output]).T
        Z = X_batch.copy()
        Z = [Z.T]
        Z = self.cell(Z)
        dZ = self.fit_func.compute_d_fitness(Z, y_batch)
        self.cell.backpropagation(dZ)
        gradients = self.cell.get_gradients()
        return gradients
