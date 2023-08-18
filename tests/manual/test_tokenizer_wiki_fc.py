from example_data import load_text
from tokenizing import tokenize, WordTextToken, EndOfSectionTextToken


print("\t".join(
    [
        token.word if isinstance(token, WordTextToken)
        else "---" if isinstance(token, EndOfSectionTextToken)
        else ""
        for token in tokenize(load_text("wikipedia-frances-cleveland.txt"))
    ]
))
