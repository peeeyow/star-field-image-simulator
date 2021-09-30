import numpy as np
import pytest

from star_field_image_simulator.image_generation.data_manipulation import Star

from numpy.random import default_rng
from pytest import approx
from .constants import REL


rng = default_rng()


@pytest.mark.parametrize(
    "index, right_ascension, declination, magnitude",
    [
        (1, 20, 20, 90),
        (2, 34.45, 23.59, 0.23),
        (
            rng.integers(0, 200_000),
            rng.uniform(0, 360),
            rng.uniform(-90, 90),
            rng.uniform(0, 30),
        ),
    ],
)
def test_star_init(index, right_ascension, declination, magnitude):
    star = Star(index, right_ascension, declination, magnitude)
    assert star.index == index
    assert star.right_ascension == approx(right_ascension, rel=REL)
    assert star.declination == approx(declination, rel=REL)
    assert star.magnitude == approx(magnitude, rel=REL)
    assert star.u is None
    assert star.v is None


@pytest.mark.parametrize(
    "index, right_ascension, declination, magnitude, X, Y, Z",
    [
        (1, 0, 0, 0, 1, 0, 0),
        (2, 30, 60, 0, (3 ** 0.5 / 4), 1 / 4, 3 ** 0.5 / 2),
        (3, 90, 0, 0, 0, 1, 0),
        (4, 0, 90, 0, 0, 0, 1),
        (5, 34.45, 23.59, 0, 0.7557087, 0.5184138, 0.4001890),
    ],
)
def test_star_cartesian(
    index, right_ascension, declination, magnitude, X, Y, Z
):
    star = Star(index, right_ascension, declination, magnitude)
    assert star.X == approx(X, rel=REL)
    assert star.Y == approx(Y, rel=REL)
    assert star.Z == approx(Z, rel=REL)


# all camera matrices uses fov = (12,12) and res = (1024, 1024)
@pytest.mark.parametrize(
    "right_ascension, declination, camera_matrix, u, v",
    [
        (
            0,
            0,
            # test (alpha0, delta0, phi0) = (1.43, -2.56, 69)
            np.array(
                [
                    [-264.696922372772, -1752.88895437964, 4566.13134910905],
                    [-475.786594589905, -4561.09536090734, -1721.12633747041],
                    [-0.998690866589801, -0.024930711441, 0.044665564108286],
                ]
            ),
            265.04390019769,
            476.410279203373,
        ),
        (
            20,
            20,
            # test (alpha0, delta0, phi0) = (20, 20, 0)
            np.array(
                [
                    [1213.9940212359, -4742.12959945932, -175.114313382742],
                    [1113.51581237692, 405.286611089947, -4752.69028476231],
                    [-0.883022221559489, -0.32139380484327, -0.342020143325],
                ]
            ),
            512,
            512,
        ),
        (
            43.29,
            -84.91,
            # test (alpha0, delta0, phi0) = (214.26, -88.18, 37)
            np.array(
                [
                    [-4598.44382202216, 1575.02714178946, 604.85003327187],
                    [1576.85796980135, 4621.15490981963, 388.182800932945],
                    [0.0262490816715987, 0.017879069553638, 0.999495535049204],
                ]
            ),
            809.462353824356,
            3.71984543373905,
        ),
    ],
)
def test_compute_pixel_coordinate(
    right_ascension, declination, camera_matrix, u, v
):
    star = Star(0, right_ascension, declination, 0)
    star.compute_pixel_coordinate(camera_matrix)
    assert star.u == approx(u, rel=REL)
    assert star.v == approx(v, rel=REL)
