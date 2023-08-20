from os.path import join as joinpath
import numpy.typing as npt
import numpy as np
from bij_map import BijMap


class ModelFilenames:
    WORD_REL_POS_UNSIGNED_FILENAME = joinpath("saved_models", "word_rel_pos_unsigned.npy")
    WORD_REL_POS_UNSIGNED_WORD_INDEXES_FILENAME = joinpath("saved_models", "word_rel_pos_unsigned_word_indexes.dat")
    WORD_REL_POS_SIGNED_FILENAME = joinpath("saved_models", "word_rel_pos_signed.npy")
    WORD_REL_POS_SIGNED_WORD_INDEXES_FILENAME = joinpath("saved_models", "word_rel_pos_signed_word_indexes.dat")
    PCA_WORD_REL_POS_UNSIGNED_FILENAME = joinpath("saved_models", "pca_word_rel_pos_unsigned.npy")
    PCA_WORD_REL_POS_SIGNED_FILENAME = joinpath("saved_models", "pca_word_rel_pos_signed.npy")


def save_matrix(filename: str, data: npt.NDArray) -> None:
    np.save(filename, data, allow_pickle=False)


def load_matrix(filename: str) -> npt.NDArray:
    data = np.load(filename, allow_pickle=False)
    assert isinstance(data, np.ndarray)
    return data


def save_word_indexes(filename: str, bm: BijMap[str, int]) -> None:
    with open(filename, "w+") as file:
        for a in bm.iterate_to():
            b = bm.get_to(a)
            assert ":" not in a, "Can't write a word with a colon in"
            file.write(f"{a}:{str(b)}\n")


def load_word_indexes(filename: str) -> BijMap[str, int]:

    bm = BijMap[str, int]()

    with open(filename, "r") as file:
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
