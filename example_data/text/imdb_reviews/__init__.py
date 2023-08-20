from typing import List, Optional
from pathlib import Path
from progress import Progress


TRAIN_DATA_POS_PATH: Path = Path(__file__).parent.absolute()/"reviews"/"training"/"pos"
TRAIN_DATA_NEG_PATH: Path = Path(__file__).parent.absolute()/"reviews"/"training"/"neg"
TEST_DATA_POS_PATH: Path = Path(__file__).parent.absolute()/"reviews"/"testing"/"pos"
TEST_DATA_NEG_PATH: Path = Path(__file__).parent.absolute()/"reviews"/"testing"/"neg"


assert TRAIN_DATA_POS_PATH.is_dir()
assert TRAIN_DATA_NEG_PATH.is_dir()
assert TEST_DATA_POS_PATH.is_dir()
assert TEST_DATA_NEG_PATH.is_dir()


def load_reviews(progress: Optional[Progress] = None) -> List[str]:
    """Loads all the reviews and returns them as a list of strings"""

    filepaths: List[Path] = list(TRAIN_DATA_POS_PATH.glob("*.txt")) \
        + list(TRAIN_DATA_NEG_PATH.glob("*.txt")) \
        + list(TEST_DATA_POS_PATH.glob("*.txt")) \
        + list(TEST_DATA_NEG_PATH.glob("*.txt"))

    texts: List[str] = []

    if progress:
        progress.max = len(filepaths)

    for path in filepaths:

        texts.append(path.read_text(encoding="UTF-8"))

        if progress:
            progress.next()

    if progress:
        progress.finish()

    return texts


def load_reviews_joined(progress: Optional[Progress] = None) -> str:
    """Loads all the reviews and joins them together into a single string, separating reviews with full stops"""

    texts = load_reviews(progress=progress)

    out = ".".join(texts)

    return out
