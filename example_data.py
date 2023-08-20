from pathlib import Path


EXAMPLE_TEXT_PATH = Path("example_data", "text")
EXAMPLE_WIKIPEDIA_TEXT_PATH = Path(EXAMPLE_TEXT_PATH, "wikipedia-articles")


def __load_file(filepath: Path) -> str:
    with filepath.open("r") as file:
        return file.read()


def load_text(filepath: Path) -> str:
    return __load_file(filepath)
