from typing import List
from abc import ABC, abstractmethod
from os.path import isfile
from tokenizing import TextToken, tokenize, WordTextToken, EndOfSectionTextToken, UnhandledTextTokenTypeException


class ITextSource(ABC):
    @abstractmethod
    def get_position(self) -> int:
        pass
    @abstractmethod
    def get_max_position(self) -> int:
        """The position of the end of the text. This position is the first invalid position greater than 0"""
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
    @abstractmethod
    def read_all_sections(self) -> List[List[WordTextToken]]:
        """Reads all sections of text delimited by a `EndOfSectionTextToken`"""
        pass


class RawTextSource(ITextSource):

    def __init__(self,
                 text: str):

        self._text = text

        self._token_list: List[TextToken] = tokenize(self._text)

        self.__pos: int = 0

    def __pos_is_valid(self, pos: int) -> bool:
        return 0 <= pos < len(self._token_list)

    def get_position(self) -> int:
        return self.__pos

    def get_max_position(self) -> int:
        return len(self._token_list)

    def read_forwards(self) -> TextToken:
        if not self.__pos_is_valid(self.__pos):
            raise IndexError()
        token = self._token_list[self.__pos]
        self.__pos += 1
        return token

    def read_backwards(self) -> TextToken:
        if not self.__pos_is_valid(self.__pos):
            raise IndexError()
        token = self._token_list[self.__pos]
        self.__pos -= 1
        return token

    def read_all(self) -> List[TextToken]:
        return self._token_list.copy()

    def read_at(self, pos: int) -> TextToken:
        if not self.__pos_is_valid(pos):
            raise IndexError()
        return self._token_list[pos]

    def read_all_sections(self) -> List[List[WordTextToken]]:

        sections: List[List[WordTextToken]] = []
        curr: List[WordTextToken] = []

        for i in range(self.get_max_position()):

            token = self.read_at(i)

            match token:
                case WordTextToken():
                    curr.append(token)
                case EndOfSectionTextToken():
                    if len(curr) > 0:
                        sections.append(curr.copy())
                        curr = []
                case _:
                    raise UnhandledTextTokenTypeException()

        if len(curr) > 0:
            sections.append(curr.copy())
            curr = []

        return sections


class FileTextSource(RawTextSource):

    def __init__(self,
                 filename: str):

        if not isfile(filename):
            raise ValueError(filename)

        self._filename = filename
        with open(self._filename, "r") as file:
            text = file.read()

        super().__init__(text)

    @property
    def filename(self) -> str:
        return self._filename
