from pathlib import Path
import numpy.typing as npt
import numpy as np
from bij_map import BijMap


class ModelFilepaths:
    WORD_REL_POS_UNSIGNED = Path("saved_models", "word_rel_pos_unsigned.npy")
    WORD_REL_POS_UNSIGNED_WORD_INDEXES = Path("saved_models", "word_rel_pos_unsigned_word_indexes.dat")
    WORD_REL_POS_SIGNED = Path("saved_models", "word_rel_pos_signed.npy")
    WORD_REL_POS_SIGNED_WORD_INDEXES = Path("saved_models", "word_rel_pos_signed_word_indexes.dat")
    PCA_WORD_REL_POS_UNSIGNED = Path("saved_models", "pca_word_rel_pos_unsigned.npy")
    PCA_WORD_REL_POS_SIGNED = Path("saved_models", "pca_word_rel_pos_signed.npy")


def save_matrix(filepath: Path, data: npt.NDArray) -> None:
    with filepath.open("wb") as file:
        np.save(file, data, allow_pickle=False)


def load_matrix(filepath: Path) -> npt.NDArray:
    with filepath.open("rb") as file:
        data = np.load(file, allow_pickle=False)
    assert isinstance(data, np.ndarray)
    return data


def save_word_indexes(filepath: Path, bm: BijMap[str, int]) -> None:
    with filepath.open("w+") as file:
        for a in bm.iterate_to():
            b = bm.get_to(a)
            assert ":" not in a, "Can't write a word with a colon in"
            file.write(f"{a}:{str(b)}\n")


def load_word_indexes(filepath: Path) -> BijMap[str, int]:

    bm = BijMap[str, int]()

    with filepath.open("r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.split(":")
        assert len(parts) == 2, "Invalid file"
        a, b_str = parts[0], parts[1]
        try:
            b = int(b_str)
        except ValueError:
            raise Exception("Invalid file")
        bm.set_to(a, b)

    return bm
