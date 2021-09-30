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
