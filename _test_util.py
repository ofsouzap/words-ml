import numpy as np
import numpy.typing as npt
from check import *


def assert_arrays_close(xs: npt.NDArray,
                        ys: npt.NDArray,
                        rtol: float = 1e-05,
                        atol: float = 1e-08) -> None:
    assert np.all(np.isclose(xs, ys, rtol=rtol, atol=atol), axis=None)


def assert_matrix_symmetric(m: npt.NDArray) -> None:

    check_is_mat(m)

    for i in range(m.shape[0]):
        for j in range(i):
            assert np.isclose(m[i,j], m[j,i])
