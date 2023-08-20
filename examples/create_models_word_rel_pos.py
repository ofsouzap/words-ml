import numpy.typing as npt
import numpy as np
from progress.bar import ShadyBar
from os.path import join as joinpath
from bij_map import BijMap
from pca import pca
from learn import learn_word_rel_pos
from text_source import RawTextSource
from example_data import load_text as load_example_text
from example_data import EXAMPLE_WIKIPEDIA_TEXT_PATH
from saved_models import save_matrix, save_word_indexes, ModelFilenames


def create_models(
    pos_mat: npt.NDArray,
    word_indexes: BijMap[str, int],
    data_filename: str,
    word_indexes_filename: str,
    pca_filename: str) -> None:

    save_word_indexes(word_indexes_filename, word_indexes)

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

    save_matrix(data_filename, normalized_pos_mat)

    mapped_mat = np.where(
        normalized_pos_mat == np.inf,
        np.zeros_like(normalized_pos_mat),
        np.exp(np.negative(normalized_pos_mat))
    )

    bar = ShadyBar("Covariance Matrix")
    reduced = pca(mapped_mat, mapped_mat.shape[0], bar)

    save_matrix(pca_filename, reduced)


def main():

    _texts = [load_example_text(joinpath(EXAMPLE_WIKIPEDIA_TEXT_PATH, fn)) for fn in [
        # "wikipedia-frances-cleveland.txt",
        "wikipedia-google.txt",
        "wikipedia-github.txt",
        "wikipedia-france.txt",
    ]]

    _fulltext = ".".join(_texts)

    text_source = RawTextSource(_fulltext)

    unsigned_pos_mat, unsigned_word_indexes = learn_word_rel_pos(text_source, max_look_dist=20, signed=False, word_count_max=1000)
    signed_pos_mat, signed_word_indexes = learn_word_rel_pos(text_source, max_look_dist=20, signed=True, word_count_max=1000)

    create_models(
        pos_mat=unsigned_pos_mat,
        word_indexes=unsigned_word_indexes,
        data_filename=ModelFilenames.WORD_REL_POS_UNSIGNED_FILENAME,
        word_indexes_filename=ModelFilenames.WORD_REL_POS_UNSIGNED_WORD_INDEXES_FILENAME,
        pca_filename=ModelFilenames.PCA_WORD_REL_POS_UNSIGNED_FILENAME
    )

    create_models(
        pos_mat=signed_pos_mat,
        word_indexes=signed_word_indexes,
        data_filename=ModelFilenames.WORD_REL_POS_SIGNED_FILENAME,
        word_indexes_filename=ModelFilenames.WORD_REL_POS_SIGNED_WORD_INDEXES_FILENAME,
        pca_filename=ModelFilenames.PCA_WORD_REL_POS_SIGNED_FILENAME
    )


if __name__ == "__main__":
    main()
