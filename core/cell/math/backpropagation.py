from abc import ABC, abstractmethod

import numpy as np


class Backpropagatable(ABC):
    @abstractmethod
    def backpropagation(self, dx: np.ndarray, meta_args=None) -> np.ndarray:
        pass

    @abstractmethod
    def forward(self, x: np.ndarray, meta_args=None) -> np.ndarray:
        pass

    @abstractmethod
    def get_gradients(self):
        pass
