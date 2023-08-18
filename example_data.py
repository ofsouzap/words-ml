from os.path import join as joinpath


EXAMPLE_TEXT_PATH = joinpath("example_data", "text")


def __load_file(filename: str) -> str:
    with open(filename, "r") as file:
        return file.read()


def load_text(name: str) -> str:
    return __load_file(joinpath(EXAMPLE_TEXT_PATH, name))
