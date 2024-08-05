import hashlib
from abc import ABC, abstractmethod
from typing import List


class HashTree(ABC):
    def __init__(self, value):
        self.value = value

    @abstractmethod
    def children(self) -> "List[HashTree]":
        pass


class HashLeaf(HashTree):
    def children(self) -> "List[HashTree]":
        return []


class HashNode(HashTree):
    def __init__(self, value, children=None):
        super().__init__(value)
        self.__children = children or []

    def children(self) -> "List[HashTree]":
        return self.__children


class Hashable(ABC):
    @abstractmethod
    def hash_tree(self) -> HashTree:
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
