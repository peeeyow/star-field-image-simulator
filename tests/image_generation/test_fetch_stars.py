import numpy as np
import pytest
import numpy.testing

from star_field_image_simulator.image_generation.data_manipulation import (
    fetch_stars,
)

from numpy.random import default_rng
from .constants import REL, DATA_PATH


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
def test_fetch_stars_north_pole(alpha0):
    actual_catalog = fetch_stars(
        alpha0, 90, 12, 12, 5.5, DATA_PATH / "sc_northpole.db"
    )
    expected_catalog = np.array(
        [
            [4, 22.5, 81.52, 4.38841501028722],
            [6, 45.0, 81.52, 2.24232748829928],
            [10, 90.0, 81.52, 2.98447455782437],
            [14, 135.0, 81.52, 2.86353789453074],
            [16, 157.5, 81.52, 5.04244982756337],
            [20, 202.5, 81.52, 5.22555027849929],
            [22, 225.0, 81.52, 2.6006729290544],
            [26, 270.0, 81.52, 0.190968558434914],
            [28, 292.5, 81.52, 1.64608407169953],
            [30, 315.0, 81.52, 3.04241416434089],
            [32, 337.5, 81.52, 2.07510087245706],
            [34, 360.0, 81.52, 3.36580061664078],
        ]
    )
    numpy.testing.assert_allclose(actual_catalog, expected_catalog, atol=REL)


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
def test_fetch_stars_south_pole(alpha0):
    actual_catalog = fetch_stars(
        alpha0, -90, 12, 12, 5.5, DATA_PATH / "sc_southpole.db"
    )
    expected_catalog = np.array(
        [
            [4, 22.5, -81.52, 0.0234277566595342],
            [12, 112.5, -81.52, 4.04387117642933],
            [14, 135.0, -81.52, 3.92737694508388],
            [20, 202.5, -81.52, 1.73780578407228],
            [24, 247.5, -81.52, 4.66173351537266],
        ]
    )
    numpy.testing.assert_allclose(actual_catalog, expected_catalog, atol=REL)


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
def test_fetch_stars_with_loop(alpha0):
    # when fov = (12, 12) RA starts to loop when delta = +-87.2980497
    actual_catalog = fetch_stars(
        alpha0, 87.3, 12, 12, 5.5, DATA_PATH / "sc_with_loop.db"
    )
    expected_catalog = np.array(
        [
            [3, 0.0, 90.0, 1.27092490352883],
            [8, 45.0, 87.31, 3.65240499644074],
            [14, 90.0, 87.31, 1.96851254024426],
            [17, 112.5, 87.31, 1.0640932963933],
            [18, 112.5, 90.0, 3.49896078874778],
            [26, 180.0, 87.31, 0.238534421342806],
            [27, 180.0, 90.0, 2.70636202427427],
            [29, 202.5, 87.31, 5.429380746765],
            [30, 202.5, 90.0, 4.99251726939971],
            [32, 225.0, 87.31, 0.615947297861928],
            [33, 225.0, 90.0, 2.87847369110188],
            [38, 270.0, 87.31, 4.62692004541187],
            [44, 315.0, 87.31, 0.233698513082187],
            [45, 315.0, 90.0, 5.44648400717444],
            [47, 337.5, 87.31, 3.06207879741723],
            [50, 360.0, 87.31, 0.506511690057861],
            [51, 360.0, 90.0, 4.67483192978273],
        ]
    )
    numpy.testing.assert_allclose(actual_catalog, expected_catalog, atol=REL)


@pytest.mark.parametrize(
    "alpha0, delta0, expected_catalog",
    [
        (
            0,
            87.29,
            np.array(
                [
                    [18, 22.5, 90.0, 0.00822666061251365],
                    [72, 157.5, 90.0, 1.38217312245601],
                    [90, 202.5, 90.0, 3.0801422702624],
                    [99, 225.0, 90.0, 0.690990552677143],
                    [108, 247.5, 90.0, 2.87436280942258],
                    [126, 292.5, 90.0, 4.79136551541631],
                    [144, 337.5, 90.0, 5.42638648227378],
                    [153, 360.0, 90.0, 5.37956078910627],
                ]
            ),
        ),
        (
            45,
            87.29,
            np.array(
                [
                    [18, 22.5, 90.0, 0.00822666061251365],
                    [72, 157.5, 90.0, 1.38217312245601],
                    [81, 180.0, 90.0, 5.34114149764781],
                    [90, 202.5, 90.0, 3.0801422702624],
                    [108, 247.5, 90.0, 2.87436280942258],
                    [126, 292.5, 90.0, 4.79136551541631],
                    [144, 337.5, 90.0, 5.42638648227378],
                    [153, 360.0, 90.0, 5.37956078910627],
                ]
            ),
        ),
        (
            90,
            87.29,
            np.array(
                [
                    [18, 22.5, 90.0, 0.00822666061251365],
                    [72, 157.5, 90.0, 1.38217312245601],
                    [81, 180.0, 90.0, 5.34114149764781],
                    [90, 202.5, 90.0, 3.0801422702624],
                    [99, 225.0, 90.0, 0.690990552677143],
                    [108, 247.5, 90.0, 2.87436280942258],
                    [126, 292.5, 90.0, 4.79136551541631],
                    [144, 337.5, 90.0, 5.42638648227378],
                    [153, 360.0, 90.0, 5.37956078910627],
                ]
            ),
        ),
    ],
)
def test_fetch_stars_with_overflow(alpha0, delta0, expected_catalog):
    actual_catalog = fetch_stars(
        alpha0, delta0, 12, 12, 5.5, DATA_PATH / "sc_no_loop.db"
    )
    numpy.testing.assert_allclose(actual_catalog, expected_catalog, atol=REL)


@pytest.mark.parametrize(
    "alpha0, delta0, expected_catalog",
    [
        (
            180,
            0,
            np.array([]),
        ),
        (
            90,
            45,
            np.array(
                [
                    [43, 90.0, 45.0, 2.37295730586114],
                ]
            ),
        ),
        (
            200,
            70,
            np.array(
                [
                    [80, 180.0, 67.5, 1.52020699338086],
                ]
            ),
        ),
    ],
)
def test_fetch_stars_no_loop(alpha0, delta0, expected_catalog):
    actual_catalog = fetch_stars(
        alpha0, delta0, 12, 12, 5.5, DATA_PATH / "sc_no_loop.db"
    )
    numpy.testing.assert_allclose(actual_catalog, expected_catalog, atol=REL)
