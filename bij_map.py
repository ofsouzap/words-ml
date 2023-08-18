from typing import Dict, TypeVar, Generic


A = TypeVar("A")
B = TypeVar("B")


class BijMap(Generic[A, B]):
    """A class for a bijective mapping"""

    def __init__(self):
        self.__to: Dict[A, B] = {}
        self.__from: Dict[B, A] = {}

    def __add(self, a: A, b: B) -> None:
        self.__to[a] = b
        self.__from[b] = a

    def set_to(self, a: A, b: B) -> None:
        self.__add(a, b)

    def set_from(self, b: B, a: A) -> None:
        self.__add(a, b)

    def get_to(self, a: A) -> B:
        if a in self.__to:
            return self.__to[a]
        else:
            raise KeyError(a)

    def get_from(self, b: B) -> A:
        if b in self.__to:
            return self.__from[b]
        else:
            raise KeyError(b)
