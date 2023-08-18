from typing import Dict, List, Tuple
import numpy as np
import pytest
from learn import learn_word_rel_pos
from text_source import RawTextSource


_CASES: List[Tuple[str, Dict[Tuple[str, str], Tuple[float, float]], int]] = [
    (
        "word more more more",
        {
            ("word", "word"): (np.inf, np.inf),
            ("word", "more"): (2, 2),
            ("more", "word"): (2, -2),
            ("more", "more"): (4/3, 0)
        },
        5
    ),
    # TODO - more test cases
]
"""Each value has the input text and a dictionary. The dictonary values are `(unsigned_expected_value, signed_expected_value)`"""


@pytest.mark.parametrize(("in_text", "exp_vals", "max_look_dist"), _CASES)
def test_unsigned_symmetry(in_text: str, exp_vals: Dict[Tuple[str, str], float], max_look_dist: int):

    source = RawTextSource(in_text)

    mat, _ = learn_word_rel_pos(source, max_look_dist=max_look_dist, signed=False)

    for i in range(mat.shape[0]):
        for j in range(i+1):
            if i == j:
                assert np.isclose(mat[i,j], mat[j,i])


@pytest.mark.parametrize(("in_text", "exp_vals", "max_look_dist"), _CASES)
def test_signed_antisymmetry(in_text: str, exp_vals: Dict[Tuple[str, str], float], max_look_dist: int):

    source = RawTextSource(in_text)

    mat, _ = learn_word_rel_pos(source, max_look_dist=max_look_dist, signed=True)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i != j:
                assert np.isclose(mat[i,j], -mat[j,i])


@pytest.mark.parametrize(("in_text", "exp_vals", "max_look_dist"), _CASES)
def test_values_unsigned(in_text: str, exp_vals: Dict[Tuple[str, str], Tuple[float, float]], max_look_dist: int):

    source = RawTextSource(in_text)

    mat, word_indexes = learn_word_rel_pos(source, max_look_dist=max_look_dist, signed=False)

    for pair in exp_vals:

        i1 = word_indexes.get_to(pair[0])
        i2 = word_indexes.get_to(pair[1])

        assert np.isclose(mat[i1,i2], exp_vals[pair][0])


@pytest.mark.parametrize(("in_text", "exp_vals", "max_look_dist"), _CASES)
def test_values_signed(in_text: str, exp_vals: Dict[Tuple[str, str], Tuple[float, float]], max_look_dist: int):

    source = RawTextSource(in_text)

    mat, word_indexes = learn_word_rel_pos(source, max_look_dist=max_look_dist, signed=True)

    for pair in exp_vals:

        i1 = word_indexes.get_to(pair[0])
        i2 = word_indexes.get_to(pair[1])

        assert np.isclose(mat[i1,i2], exp_vals[pair][1])
