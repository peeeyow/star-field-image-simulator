import numpy as np
import numpy.typing as npt

from numpy.random import default_rng
from typing import Union

rng = default_rng()


def add_dark_current_noise(
    image: npt.NDArray[Union[np.uint8, np.float64]], nDC: float, tauDC: float
):
    if nDC == 0 or tauDC == 0:
        return image

    shape = image.shape
    return image + nDC * tauDC + np.sqrt(nDC * tauDC) * rng.normal(0, 1, shape)


def add_shot_noise(
    image: npt.NDArray[Union[np.uint8, np.float64]], varNoise: float
):
    if varNoise == 0:
        return image

    shape = image.shape
    return image + np.sqrt(image) * rng.normal(0, varNoise, shape)


def add_read_noise(
    image: npt.NDArray[Union[np.uint8, np.float64]], nRN: float
):
    return image + nRN
