import numpy
import pytest
import numpy.testing

from star_field_image_simulator.image_generation.data_manipulation import (
    Celestial2Image,
)

from numpy.random import default_rng
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
    "alpha0, delta0, phi0, rotation_matrix",
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
def test_rotation_matrix(alpha0, delta0, phi0, rotation_matrix):
    c2i = Celestial2Image(alpha0, delta0, phi0, 0, 0, 0, 0)
    numpy.testing.assert_allclose(
        c2i.rotation_matrix, rotation_matrix, atol=REL
    )


@pytest.mark.parametrize(
    "fovX, fovY, resX, resY, projection_matrix",
    [
        (
            8,
            8,
            512,
            512,
            numpy.array(
                [[3660.970562, 0, 256], [0, -3660.970562, 256], [0, 0, 1]]
            ),
        ),
        (
            12,
            12,
            512,
            512,
            numpy.array(
                [
                    [2435.6773, 0, 256],
                    [0, -2435.6773, 256],
                    [0, 0, 1],
                ]
            ),
        ),
        (
            8,
            8,
            1024,
            1024,
            numpy.array(
                [
                    [7321.941123, 0, 512],
                    [0, -7321.941123, 512],
                    [0, 0, 1],
                ]
            ),
        ),
        (
            12,
            12,
            1024,
            1024,
            numpy.array(
                [
                    [4871.354601, 0, 512],
                    [0, -4871.354601, 512],
                    [0, 0, 1],
                ]
            ),
        ),
    ],
)
def test_projection_matrix(fovX, fovY, resX, resY, projection_matrix):
    c2i = Celestial2Image(0, 0, 0, fovX, fovY, resX, resY)
    numpy.testing.assert_allclose(
        c2i.projection_matrix, projection_matrix, atol=REL
    )


@pytest.mark.parametrize(
    "alpha0, delta0, phi0, fovX, fovY, resX, resY, camera_matrix",
    [
        (
            90,
            90,
            90,
            8,
            8,
            512,
            512,
            numpy.array(
                [
                    [0, -3660.970562, -256],
                    [3660.970562, 0, -256],
                    [0, 0, -1],
                ]
            ),
        ),
        (
            90,
            90,
            90,
            12,
            12,
            1024,
            1024,
            numpy.array(
                [
                    [0, -4871.354601, -512],
                    [4871.354601, 0, -512],
                    [0, 0, -1],
                ]
            ),
        ),
        (
            227.6493,
            -72.4427,
            -39.8703,
            8,
            8,
            512,
            512,
            numpy.array(
                [
                    [-5.171476e02, 3.603556e03, -4.638817e02],
                    [3.591127e03, 4.558595e02, -6.035240e02],
                    [2.032179e-01, 2.229370e-01, 9.534157e-01],
                ]
            ),
        ),
        (
            227.6493,
            -72.4427,
            -39.8703,
            12,
            12,
            1024,
            1024,
            numpy.array(
                [
                    [-6.533024e02, 4.833160e03, -4.538705e02],
                    [4.813243e03, 6.447779e02, -6.396812e02],
                    [2.032179e-01, 2.229370e-01, 9.534157e-01],
                ]
            ),
        ),
    ],
)
def test_camera_matrix(
    alpha0, delta0, phi0, fovX, fovY, resX, resY, camera_matrix
):
    c2i = Celestial2Image(alpha0, delta0, phi0, fovX, fovY, resX, resY)
    numpy.testing.assert_allclose(c2i.camera_matrix, camera_matrix, atol=REL)


def test_camera_matrix_on_the_fly():
    alpha0 = 350
    delta0 = 90
    phi0 = 0
    fovX = 12
    fovY = 12
    resX = 1024
    resY = 1024
    c2i = Celestial2Image(alpha0, delta0, phi0, fovX, fovY, resX, resY)
    camera_matrix = numpy.array(
        [
            [-845.901849157002, -4797.34777830511, -512],
            [4797.34777830511, -845.901849157002, -512],
            [0, 0, -1],
        ]
    )
    numpy.testing.assert_allclose(c2i.camera_matrix, camera_matrix, atol=REL)
