import pytest

from star_field_image_simulator.image_generation.data_manipulation import (
    create_false_stars,
    remove_random_stars,
    Star,
)

from numpy.random import default_rng

rng = default_rng()


@pytest.mark.parametrize(
    "num_missing_stars, num_remaining_stars",
    [
        (0, 8),
        (1, 7),
        (2, 6),
        (3, 5),
        (4, 4),
        (5, 3),
        (6, 3),
        (7, 3),
        (8, 3),
    ],
)
def test_remove_random_stars(num_missing_stars, num_remaining_stars):
    stars = [
        Star(18, 22.5, 90.0, 0.00822666061251365),
        Star(72, 157.5, 90.0, 1.38217312245601),
        Star(90, 202.5, 90.0, 3.0801422702624),
        Star(99, 225.0, 90.0, 0.690990552677143),
        Star(108, 247.5, 90.0, 2.87436280942258),
        Star(126, 292.5, 90.0, 4.79136551541631),
        Star(144, 337.5, 90.0, 5.42638648227378),
        Star(153, 360.0, 90.0, 5.37956078910627),
    ]
    stars = remove_random_stars(stars, num_missing_stars)
    assert len(stars) == num_remaining_stars


def test_add_false_star():
    num_false_stars = 4
    resX = 1024
    resY = 1024
    false_mag = 5.5
    false_stars = create_false_stars(num_false_stars, resX, resY, false_mag)
    assert len(false_stars) == num_false_stars
