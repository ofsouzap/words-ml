from os.path import join as joinpath


EXAMPLE_TEXT_PATH = joinpath("example_data", "text")
EXAMPLE_WIKIPEDIA_TEXT_PATH = joinpath(EXAMPLE_TEXT_PATH, "wikipedia-articles")


def __load_file(filename: str) -> str:
    with open(filename, "r") as file:
        return file.read()


def load_text(filename: str) -> str:
    return __load_file(filename)
