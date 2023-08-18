from typing import Union
import numpy as np
from numpy import typing as npt


DEBUG_MODE = True
"""Whether debugging is currently enabled"""


def enable_debug() -> None:
    global DEBUG_MODE
    DEBUG_MODE = True


def disable_debug() -> None:
    global DEBUG_MODE
    DEBUG_MODE = False


def check(cond: Union[bool, np.bool_], message: str = "") -> None:
    if DEBUG_MODE:
        assert cond, message


def check_mat_dim(m: npt.NDArray, n: int, message: str = ""): check(m.shape[0] == m.shape[1] == n, message)
def check_is_mat(a: npt.NDArray, message: str = ""): check(a.ndim == 2, message)
def check_mat_square(m: npt.NDArray, message: str = ""): check(m.shape[0] == m.shape[1], message)
def check_mat_symmetrical(m: npt.NDArray, message: str = ""): check(np.all(np.isclose(m, m.T), axis=None), message)
