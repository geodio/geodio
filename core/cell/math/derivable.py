from typing import Optional


class Derivable:
    def d(self, dx) -> 'Optional[Derivable]':
        return None


class WeightDerivable:
    def d_w(self, dw) -> 'Optional[Derivable]':
        return None
