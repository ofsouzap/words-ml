from typing import List, Set
from abc import ABC
import re


__WORD_CHARACTERS: Set[str] = {
    *[chr(x) for x in range(ord("A"), ord("Z")+1)],
    *[chr(x) for x in range(ord("a"), ord("z")+1)],
}
"""Characters that are accepted as being parts of a word"""

__WORD_SEPARATORS: Set[str] = { " ", "-", "," }
"""Characters that are accepted as separating words."""


__SECTION_SEPARATORS: Set[str] = { ".", "\n" }
"""Characters that are accepted as separating sections of text."""


class TextToken(ABC):
    """A piece of a tokenized text"""
    pass


class WordTextToken(TextToken):

    _VALID_REGEX = re.compile(r"\w+", flags=re.IGNORECASE)

    def __init__(self, word: str):

        if not WordTextToken._VALID_REGEX.fullmatch(word):
            raise ValueError(word)

        self.word = word


class EndOfSectionTextToken(TextToken):
    pass


def tokenize(text: str) -> List[TextToken]:

    tokens: List[TextToken] = []
    curr: str = ""

    for char in text:

        if char in __WORD_CHARACTERS:

            curr += char

        elif char in __WORD_SEPARATORS:

            if curr:
                tokens.append(WordTextToken(curr))
                curr = ""

        elif char in __SECTION_SEPARATORS:

            if curr:
                tokens.append(WordTextToken(curr))
                curr = ""

            if (len(tokens) > 0) and (not isinstance(tokens[-1], EndOfSectionTextToken)):
                tokens.append(EndOfSectionTextToken())

    return tokens
