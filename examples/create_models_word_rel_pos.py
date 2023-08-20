import numpy.typing as npt
import numpy as np
from pathlib import Path
from progress.bar import IncrementalBar
from bij_map import BijMap
from pca import covariance_matrix, principal_components, project_to_components
from learn import learn_word_rel_pos
from text_source import RawTextSource
from example_data.text.wikipedia_articles import load_text as load_wikipedia_text
from example_data.text.imdb_reviews import load_reviews_joined as load_imdb_reviews_text
from saved_models import save_matrix, save_word_indexes, ModelFilepaths


def create_models(
    pos_mat: npt.NDArray,
    word_indexes: BijMap[str, int],
    data_filepath: Path,
    word_indexes_filepath: Path,
    principal_components_filepath: Path,
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

    cov_mat_bar = IncrementalBar("Covariance Matrix")
    cov_mat = covariance_matrix(mapped_mat, progress=cov_mat_bar)

    prin_comps = principal_components(cov_mat)
    save_matrix(principal_components_filepath, prin_comps)

    projected = project_to_components(mapped_mat, prin_comps)
    save_matrix(pca_filepath, projected)


def main():

    # _texts = [load_wikipedia_text(fn) for fn in [
    #     "frances-cleveland",
    #     "google",
    #     "github",
    #     "france",
    # ]]

    # _fulltext = ".".join(_texts)

    text_load_progress = IncrementalBar("Loading text data")
    _fulltext = load_imdb_reviews_text(progress=text_load_progress)[:1_000_000]

    tokenize_progress = IncrementalBar("Tokenize data")
    text_source = RawTextSource(_fulltext, tokenize_progress=tokenize_progress)

    unsigned_learn_bar = IncrementalBar("Train unsigned")
    unsigned_pos_mat, unsigned_word_indexes = learn_word_rel_pos(text_source, max_look_dist=20, signed=False, word_count_max=1000, progress=unsigned_learn_bar)

    signed_learn_bar = IncrementalBar("Train signed")
    signed_pos_mat, signed_word_indexes = learn_word_rel_pos(text_source, max_look_dist=20, signed=True, word_count_max=1000, progress=signed_learn_bar)

    create_models(
        pos_mat=unsigned_pos_mat,
        word_indexes=unsigned_word_indexes,
        data_filepath=ModelFilepaths.WORD_REL_POS_UNSIGNED,
        word_indexes_filepath=ModelFilepaths.WORD_REL_POS_UNSIGNED_WORD_INDEXES,
        principal_components_filepath=ModelFilepaths.PRINCIPAL_COMPONENTS_WORD_REL_POS_UNSIGNED,
        pca_filepath=ModelFilepaths.PCA_WORD_REL_POS_UNSIGNED
    )

    create_models(
        pos_mat=signed_pos_mat,
        word_indexes=signed_word_indexes,
        data_filepath=ModelFilepaths.WORD_REL_POS_SIGNED,
        word_indexes_filepath=ModelFilepaths.WORD_REL_POS_SIGNED_WORD_INDEXES,
        principal_components_filepath=ModelFilepaths.PRINCIPAL_COMPONENTS_WORD_REL_POS_SIGNED,
        pca_filepath=ModelFilepaths.PCA_WORD_REL_POS_SIGNED
    )


if __name__ == "__main__":
    main()
