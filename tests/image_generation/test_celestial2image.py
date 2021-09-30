import numpy
import pytest
import numpy.testing

from star_field_image_simulator.image_generation.data_manipulation import (
    Celestial2Image,
)

from numpy.random import default_rng
from pytest import approx
from .constants import REL


rng = default_rng()


@pytest.mark.parametrize(
    "alpha0",
    [
        0,
        180,
        360,
        rng.uniform(0, 360),
    ],
)
@pytest.mark.parametrize(
    "delta0",
    [
        -90,
        -45,
        0,
        45,
        90,
        rng.uniform(0, 360),
    ],
)
@pytest.mark.parametrize(
    "phi0",
    [
        -90,
        -45,
        0,
        45,
        90,
        rng.uniform(0, 360),
    ],
)
@pytest.mark.parametrize("fovX", [12])
@pytest.mark.parametrize("fovY", [12])
@pytest.mark.parametrize("resX", [1024])
@pytest.mark.parametrize("resY", [1024])
def test_cwlwstial2image_init(alpha0, delta0, phi0, fovX, fovY, resX, resY):
    c2i = Celestial2Image(alpha0, delta0, phi0, fovX, fovY, resX, resY)
    assert c2i.alpha0 == alpha0
    assert c2i.delta0 == delta0
    assert c2i.phi0 == phi0
    assert c2i.fovX == fovX
    assert c2i.fovY == fovY
    assert c2i.resX == resX
    assert c2i.resY == resY


@pytest.mark.parametrize(
    "alpha0, delta0, phi0, rot_matrix",
    [
        (0, 0, 0, numpy.array([[0, -1, 0], [0, 0, 1], [-1, 0, 0]])),
        (
            30,
            60,
            90,
            numpy.array(
                [
                    [-0.7500, -0.4330, 0.5000],
                    [-0.5000, 0.8660, 0],
                    [-0.4330, -0.2500, -0.8660],
                ]
            ),
        ),
        (90, 90, 90, numpy.array([[0, -1, 0], [-1, 0, 0], [0, 0, -1]])),
        (
            227.6493,
            -72.4427,
            -39.8703,
            numpy.array(
                [
                    [-0.1555, 0.9687, -0.1934],
                    [-0.9667, -0.1089, 0.2315],
                    [0.2032, 0.2229, 0.9534],
                ]
            ),
        ),
        (
            196.8773,
            82.3512,
            83.6799,
            numpy.array(
                [
                    [0.9107, 0.3913, 0.1323],
                    [0.3930, -0.9194, 0.0147],
                    [0.1274, 0.0386, -0.9911],
                ]
            ),
        ),
        (
            56.7407,
            84.7067,
            82.2901,
            numpy.array(
                [
                    [-0.4290, -0.8987, 0.0914],
                    [-0.9019, 0.4318, 0.0124],
                    [-0.0506, -0.0771, -0.9957],
                ]
            ),
        ),
    ],
)
def test_rot_matrix(alpha0, delta0, phi0, rot_matrix):
    c2i = Celestial2Image(alpha0, delta0, phi0, 0, 0, 0, 0)
    numpy.testing.assert_allclose(c2i.rot_matrix, rot_matrix, atol=REL)
