import numpy.typing as npt
import numpy as np
from pathlib import Path
from progress.bar import ShadyBar
from bij_map import BijMap
from pca import pca
from learn import learn_word_rel_pos
from text_source import RawTextSource
from example_data.text.wikipedia_articles import load_text as load_example_text
from saved_models import save_matrix, save_word_indexes, ModelFilepaths


def create_models(
    pos_mat: npt.NDArray,
    word_indexes: BijMap[str, int],
    data_filepath: Path,
    word_indexes_filepath: Path,
    pca_filepath: Path) -> None:

    save_word_indexes(word_indexes_filepath, word_indexes)

    offset_pos_mat = pos_mat-np.mean(np.where(
        pos_mat == np.inf,
        0,
        pos_mat
    ))

    normalization_factor = np.max(np.where(
        pos_mat == np.inf,
        0,
        np.abs(offset_pos_mat)
    ))
    normalized_pos_mat = offset_pos_mat/normalization_factor

    save_matrix(data_filepath, normalized_pos_mat)

    mapped_mat = np.where(
        normalized_pos_mat == np.inf,
        np.zeros_like(normalized_pos_mat),
        np.exp(np.negative(normalized_pos_mat))
    )

    bar = ShadyBar("Covariance Matrix")
    reduced = pca(mapped_mat, mapped_mat.shape[0], bar)

    save_matrix(pca_filepath, reduced)


def main():

    _texts = [load_example_text(fn) for fn in [
        # "frances-cleveland",
        "google",
        "github",
        "france",
    ]]

    _fulltext = ".".join(_texts)

    text_source = RawTextSource(_fulltext)

    unsigned_pos_mat, unsigned_word_indexes = learn_word_rel_pos(text_source, max_look_dist=20, signed=False, word_count_max=1000)
    signed_pos_mat, signed_word_indexes = learn_word_rel_pos(text_source, max_look_dist=20, signed=True, word_count_max=1000)

    create_models(
        pos_mat=unsigned_pos_mat,
        word_indexes=unsigned_word_indexes,
        data_filepath=ModelFilepaths.WORD_REL_POS_UNSIGNED,
        word_indexes_filepath=ModelFilepaths.WORD_REL_POS_UNSIGNED_WORD_INDEXES,
        pca_filepath=ModelFilepaths.PCA_WORD_REL_POS_UNSIGNED
    )

    create_models(
        pos_mat=signed_pos_mat,
        word_indexes=signed_word_indexes,
        data_filepath=ModelFilepaths.WORD_REL_POS_SIGNED,
        word_indexes_filepath=ModelFilepaths.WORD_REL_POS_SIGNED_WORD_INDEXES,
        pca_filepath=ModelFilepaths.PCA_WORD_REL_POS_SIGNED
    )


if __name__ == "__main__":
    main()
