from typing import Optional

import numpy as np


class Derivable:
    def __call__(self, args, meta_args=None):
        raise NotImplementedError

    def d(self, dx) -> 'Optional[Derivable]':
        return None


class WeightDerivable:
    def __call__(self, args, meta_args=None):
        raise NotImplementedError

    def d_w(self, dw) -> 'Optional[WeightDerivable]':
        return None
