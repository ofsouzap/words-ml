from typing import List
from abc import ABC, abstractmethod
from os.path import isfile

from tokenizing import TextToken, tokenize


class ITextSource(ABC):
    @abstractmethod
    def get_position(self) -> int:
        pass
    @abstractmethod
    def read_at(self, pos: int) -> TextToken:
        pass
    @abstractmethod
    def read_forwards(self) -> TextToken:
        pass
    @abstractmethod
    def read_backwards(self) -> TextToken:
        pass
    @abstractmethod
    def read_all(self) -> List[TextToken]:
        pass


class FileTextSource(ITextSource):

    def __init__(self,
                 filename: str):

        if not isfile(filename):
            raise ValueError(filename)

        self._filename = filename
        with open(self._filename, "r") as file:
            self._text = file.read()

        self._token_list: List[TextToken] = tokenize(self._text)

        self.__pos: int = 0

    @property
    def filename(self) -> str:
        return self._filename

    def get_position(self) -> int:
        return self.__pos

    def read_forwards(self) -> TextToken:
        token = self._token_list[self.__pos]
        self.__pos += 1
        return token

    def read_backwards(self) -> TextToken:
        token = self._token_list[self.__pos]
        self.__pos -= 1
        return token

    def read_all(self) -> List[TextToken]:
        return self._token_list.copy()

    def read_at(self, pos: int) -> TextToken:
        return self._token_list[pos]
