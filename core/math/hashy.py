import hashlib
from abc import ABC, abstractmethod
from typing import List


class HashTree:
    def __init__(self):
        pass


class Hashable(ABC):
    @abstractmethod
    def hash_tree(self):
        pass

    @abstractmethod
    def hash_str(self) -> str:
        pass

    def __hash__(self):
        return int(self.hash_str().encode())


def hash_with_children(value, children: List[Hashable]):
    hasher = hashlib.sha256()
    hasher.update(value.encode())
    for child in children:
        hasher.update(child.hash_str().encode())
    return hasher.hexdigest()
