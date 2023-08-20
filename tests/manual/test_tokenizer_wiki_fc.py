from example_data import load_text, EXAMPLE_WIKIPEDIA_TEXT_PATH
from os.path import join as joinpath
from tokenizing import tokenize, WordTextToken, EndOfSectionTextToken


print("\t".join(
    [
        token.word if isinstance(token, WordTextToken)
        else "---" if isinstance(token, EndOfSectionTextToken)
        else ""
        for token in tokenize(load_text(joinpath(EXAMPLE_WIKIPEDIA_TEXT_PATH, "wikipedia-frances-cleveland.txt")))
    ]
))
