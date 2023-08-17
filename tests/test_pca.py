import pytest
import numpy as np
from _test_util import *
from pca import pca


_CASES = [
    (
        np.array([
            [2],
            [4]
        ], dtype=np.float32),
        1,
        np.array([
            [2],
            [4]
        ], dtype=np.float32),
    ),
    (
        np.array([
            [1, 2],
            [4, 5],
            [1, 4]
        ], dtype=np.float32),
        2,
        np.array([
            [ -2.22850063, 0.18380681 ],
            [ -6.35008585, -0.82244131 ],
            [ -3.93777841, 1.22225251 ],
        ], dtype=np.float32),
    ),
    (
        np.array([
            [1, 2, 3],
            [4, 5, 6],
            [1, 4, 6]
        ], dtype=np.float32),
        3,
        np.array([
            [ -3.72968284, 0.27372676, 0.12058035 ],
            [ -8.67658618, -1.31014931, -0.01900461 ],
            [ -7.13407096, 1.45032332, -0.03992554 ],
        ], dtype=np.float32),
    ),
    (
        np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]
        ], dtype=np.float32),
        3,
        np.array([
            [ -3.68247103, 0.6628779, 0 ],
            [ -5.37665237, 0.30267021, 0 ],
            [ -7.07083371, -0.05753748, 0 ],
            [ -8.76501505, -0.41774517, 0 ],
        ], dtype=np.float32),
    ),
    (
        np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]
        ], dtype=np.float32),
        2,
        np.array([
            [ -3.68247103, 0.6628779 ],
            [ -5.37665237, 0.30267021 ],
            [ -7.07083371, -0.05753748 ],
            [ -8.76501505, -0.41774517 ],
        ], dtype=np.float32),
    ),
    (
        np.array([
            [1, 2, 3],
            [2, 3, 4],
            [3, 4, 5],
            [4, 5, 6]
        ], dtype=np.float32),
        1,
        np.array([
            [ -3.68247103 ],
            [ -5.37665237 ],
            [ -7.07083371 ],
            [ -8.76501505 ],
        ], dtype=np.float32),
    ),
    (
        np.array([
            [ -1.4, -6.3,  3.3 ],
            [ -4.2,  5.0,  6.0 ],
            [  0.0, 10.0,  0   ],
            [  1.1,  4.2,  6.6 ]
        ], dtype=np.float32),
        3,
        np.array([
            [ 4.641526658, 5.562314037, 0.238523051 ],
            [ -7.091668042, 4.769943814, 2.365561414 ],
            [ 50, 4.769943814, 2.365561414 ],
            [ 55.98, 4.769943814, 2.365561414 ],
        ], dtype=np.float32),
    ),
    (
        np.array([
            [ -1.4, -6.3,  3.3 ],
            [ -4.2,  5.0,  6.0 ],
            [  0.0, 10.0,  0   ],
            [  1.1,  4.2,  6.6 ]
        ], dtype=np.float32),
        2,
        np.array([
            [ 4.641526658, 5.562314037 ],
            [ -7.091668042, 4.769943814 ],
            [ 50, 4.769943814 ],
            [ 55.98, 4.769943814 ],
        ], dtype=np.float32),
    ),
    (
        np.array([
            [ -1.4, -6.3,  3.3 ],
            [ -4.2,  5.0,  6.0 ],
            [  0.0, 10.0,  0   ],
            [  1.1,  4.2,  6.6 ]
        ], dtype=np.float32),
        1,
        np.array([
            [ 4.641526658 ],
            [ -7.091668042 ],
            [ 50 ],
            [ 55.98 ],
        ], dtype=np.float32),
    ),
]


@pytest.mark.parametrize(("xs", "N", "ys"), _CASES)
def test_dimensions(xs: npt.NDArray, N: int, ys: npt.NDArray):
    out = pca(xs, N)
    assert out.ndim == 2
    assert out.shape[0] == xs.shape[0]
    assert out.shape[1] == N


@pytest.mark.parametrize(("xs", "N", "ys"), _CASES)
def test_values(xs: npt.NDArray, N: int, ys: npt.NDArray):
    out = pca(xs, N)
    facs = ys[:,0] / out[:,0]
    fac_out = out * (facs[:,np.newaxis])
    assert_arrays_close(fac_out, ys, atol=1e-5)
