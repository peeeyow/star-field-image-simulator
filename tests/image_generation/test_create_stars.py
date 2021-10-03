import pytest
from star_field_image_simulator.image_generation.constants import (
    U_COORDINATE_ORIGIN,
    V_COORDINATE_ORIGIN,
)

from star_field_image_simulator.image_generation.data_manipulation import (
    Celestial2Image,
    create_stars,
    is_within_canvass,
    Star,
)

from numpy.random import default_rng
from .constants import DATA_PATH

rng = default_rng()


@pytest.mark.parametrize(
    "star",
    [
        # all of this are located within alpha0,
        # delta0, phi0 = (350, 81.5, 0)
        Star(4, 22.5, 81.52, 4.38841501028722),
        Star(30, 315.0, 81.52, 3.04241416434089),
        Star(32, 337.5, 81.52, 2.07510087245706),
        Star(34, 360.0, 81.52, 3.36580061664078),
    ],
)
def test_is_within_canvass_true(
    star,
):
    alpha0 = 350
    delta0 = 81.5
    phi0 = 0
    fovX = 12
    fovY = 12
    u_coordinate_origin = U_COORDINATE_ORIGIN
    resX = 1024
    v_coordinate_origin = V_COORDINATE_ORIGIN
    resY = 1024
    c2i = Celestial2Image(alpha0, delta0, phi0, fovX, fovY, resX, resY)
    star.compute_pixel_coordinate(c2i.camera_matrix)
    assert is_within_canvass(
        star, u_coordinate_origin, resX, v_coordinate_origin, resY
    )


@pytest.mark.parametrize(
    "star",
    [
        Star(4, 22.5, 81.52, 4.38841501028722),
        Star(6, 45.0, 81.52, 2.24232748829928),
        Star(30, 315.0, 81.52, 3.04241416434089),
        Star(32, 337.5, 81.52, 2.07510087245706),
        Star(34, 360.0, 81.52, 3.36580061664078),
    ],
)
def test_is_within_canvass_false(star):
    alpha0 = 350
    delta0 = 90
    phi0 = 0
    fovX = 12
    fovY = 12
    u_coordinate_origin = U_COORDINATE_ORIGIN
    resX = 1024
    v_coordinate_origin = V_COORDINATE_ORIGIN
    resY = 1024
    c2i = Celestial2Image(alpha0, delta0, phi0, fovX, fovY, resX, resY)
    star.compute_pixel_coordinate(c2i.camera_matrix)
    assert not is_within_canvass(
        star, u_coordinate_origin, resX, v_coordinate_origin, resY
    )


@pytest.mark.parametrize(
    "alpha0, delta0, expected_stars",
    [
        (
            0,
            87.29,
            [
                Star(18, 22.5, 90.0, 0.00822666061251365),
                Star(72, 157.5, 90.0, 1.38217312245601),
                Star(90, 202.5, 90.0, 3.0801422702624),
                Star(99, 225.0, 90.0, 0.690990552677143),
                Star(108, 247.5, 90.0, 2.87436280942258),
                Star(126, 292.5, 90.0, 4.79136551541631),
                Star(144, 337.5, 90.0, 5.42638648227378),
                Star(153, 360.0, 90.0, 5.37956078910627),
            ],
        ),
        (180, 0, []),
        (
            90,
            45,
            [
                Star(43, 90.0, 45.0, 2.37295730586114),
            ],
        ),
        (200, 70, []),
    ],
)
def test_build_stars(alpha0, delta0, expected_stars):
    phi0 = 0
    magnitude = 5.5
    fovX = 12
    fovY = 12
    u_coordinate_origin = U_COORDINATE_ORIGIN
    resX = 1024
    v_coordinate_origin = V_COORDINATE_ORIGIN
    resY = 1024
    c2i = Celestial2Image(alpha0, delta0, phi0, fovX, fovY, resX, resY)
    actual_stars = create_stars(
        alpha0,
        delta0,
        magnitude,
        fovX,
        fovY,
        u_coordinate_origin,
        resX,
        v_coordinate_origin,
        resY,
        c2i,
        DATA_PATH / "sc_no_loop.db",
    )
    assert actual_stars == expected_stars
