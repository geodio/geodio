from abc import ABC, abstractmethod

import numpy as np


class Backpropagatable(ABC):
    @abstractmethod
    def backpropagation(self, dx: np.ndarray, meta_args=None) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray, meta_args=None) -> np.ndarray:
        pass

    def get_gradients(self):
        return self.get_local_gradients()

    @abstractmethod
    def get_local_gradients(self) -> list:
        pass
