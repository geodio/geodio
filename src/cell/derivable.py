from typing import Optional


class Derivable:
    def d(self, dx) -> 'Optional[Derivable]':
        return None
