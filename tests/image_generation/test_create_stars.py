import pytest

from star_field_image_simulator.image_generation.data_manipulation import (
    create_stars,
    Star,
)

from numpy.random import default_rng
from .constants import DATA_PATH

rng = default_rng()


@pytest.mark.parametrize(
    "alpha0",
    [
        0.0,
        45.0,
        90.0,
        135.0,
        180.0,
        225.0,
        270.0,
        315.0,
        360.0,
        rng.random() * 360,
    ],
)
def test_build_stars(alpha0):
    actual_stars = create_stars(
        alpha0, 90, 12, 12, 5.5, DATA_PATH / "sc_northpole.db"
    )
    expected_stars = [
        Star(4, 22.5, 81.52, 4.38841501028722),
        Star(6, 45.0, 81.52, 2.24232748829928),
        Star(10, 90.0, 81.52, 2.98447455782437),
        Star(14, 135.0, 81.52, 2.86353789453074),
        Star(16, 157.5, 81.52, 5.04244982756337),
        Star(20, 202.5, 81.52, 5.22555027849929),
        Star(22, 225.0, 81.52, 2.6006729290544),
        Star(26, 270.0, 81.52, 0.190968558434914),
        Star(28, 292.5, 81.52, 1.64608407169953),
        Star(30, 315.0, 81.52, 3.04241416434089),
        Star(32, 337.5, 81.52, 2.07510087245706),
        Star(34, 360.0, 81.52, 3.36580061664078),
    ]
    assert actual_stars == expected_stars


@pytest.mark.parametrize(
    "alpha0, delta0, expected_stars",
    [
        (180, 0, []),
        (
            90,
            45,
            [
                Star(43, 90.0, 45.0, 2.37295730586114),
            ],
        ),
        (
            200,
            70,
            [
                Star(80, 180.0, 67.5, 1.52020699338086),
            ],
        ),
    ],
)
def test_fetch_stars_no_loop(alpha0, delta0, expected_stars):
    actual_stars = create_stars(
        alpha0, delta0, 12, 12, 5.5, DATA_PATH / "sc_no_loop.db"
    )
    assert actual_stars == expected_stars
