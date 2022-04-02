import numpy as np
import numpy.typing as npt

from .constants import (
    DATABASE_PATH,
    SUB_IMAGE_SIZE,
    U_COORDINATE_ORIGIN,
    V_COORDINATE_ORIGIN,
    STAR_INTENSITY_LEVEL,
)
from .data_manipulation import (
    Celestial2Image,
    Star,
    create_false_stars,
    create_stars_list,
    remove_random_stars,
)
from numpy.random import default_rng
from scipy.special import erf


rng = default_rng()


def draw_star_field_image(
    stars: list[Star],
    resX: int,
    resY: int,
    star_intensity: float,
    star_sigma: float,
    integrated: bool = True,
    lazy: bool = True,
):
    indices_u, indices_v = np.meshgrid(np.arange(resX), np.arange(resY))

    star_field_image = np.zeros([resY, resX])

    sub_image_half_size = (SUB_IMAGE_SIZE - 1) // 2

    centroids = []

    for star in stars:
        Mi = star.magnitude
        Ui = star.u
        Vi = star.v
        IDi = star.index

        centroids.append((IDi, Ui, Vi))

        if lazy:
            # compute center pixel
            x_center = round(Ui)  # type: ignore
            y_center = round(Vi)  # type: ignore

            # get index of the subimage
            uScope = np.arange(
                max(x_center - sub_image_half_size, U_COORDINATE_ORIGIN),
                min(x_center + sub_image_half_size, resX),
            )
            vScope = np.arange(
                max(y_center - sub_image_half_size, V_COORDINATE_ORIGIN),
                min(y_center + sub_image_half_size, resY),
            )
        else:
            # use the whole resolution
            uScope = np.arange(U_COORDINATE_ORIGIN, resX)
            vScope = np.arange(V_COORDINATE_ORIGIN, resY)

        scope = np.ix_(vScope.astype(int), uScope.astype(int))  # type: ignore
        if integrated:
            # input to erf
            x1 = (indices_u[scope] + 1 - Ui) / (np.sqrt(2) * star_sigma)
            x2 = (indices_u[scope] - Ui) / (np.sqrt(2) * star_sigma)
            y1 = (indices_v[scope] + 1 - Vi) / (np.sqrt(2) * star_sigma)
            y2 = (indices_v[scope] - Vi) / (np.sqrt(2) * star_sigma)

            # computing starContribution
            starContribution = (
                (star_intensity / STAR_INTENSITY_LEVEL ** Mi)
                * (np.pi * star_sigma ** 2 / 2)
                * (erf(x1) - erf(x2))
                * (erf(y1) - erf(y2))
            )

            star_field_image[scope] += starContribution
        else:

            x = indices_u[scope] - Ui
            y = indices_v[scope] - Vi

            # computing starContribution
            starContribution = (
                star_intensity / STAR_INTENSITY_LEVEL ** Mi
            ) * (np.exp(-(x ** 2 + y ** 2) / (2 * star_sigma ** 2)))

            star_field_image[scope] += starContribution

    return star_field_image, centroids


def generate_star_field_image(
    alpha0: float,
    delta0: float,
    phi0: float,
    resX: int,
    resY: int,
    fovX: float,
    fovY: float,
    magnitude_limit: float,
    num_missing_stars: int,
    num_false_stars: int,
    min_false_star_magnitude: float,
    star_intensity: float,
    star_sigma: float,
    position_noise: float,
    integrated: bool = True,
    lazy: bool = True,
) -> npt.ArrayLike:
    c2i = Celestial2Image(alpha0, delta0, phi0, fovX, fovY, resX, resY)
    stars = create_stars_list(
        alpha0,
        delta0,
        magnitude_limit,
        fovX,
        fovY,
        U_COORDINATE_ORIGIN,
        resX,
        V_COORDINATE_ORIGIN,
        resY,
        c2i,
        DATABASE_PATH,
    )
    stars = remove_random_stars(stars, num_missing_stars)
    stars.extend(
        create_false_stars(
            num_false_stars, resX, resY, min_false_star_magnitude
        )
    )

    if position_noise:
        for star in stars:
            pixels_u = rng.normal(0, position_noise)
            pixels_v = rng.normal(0, position_noise)
            star.u = np.clip(star.u + pixels_u, 0, resX)  # type: ignore
            star.v = np.clip(star.v + pixels_v, 0, resY)  # type: ignore

    return draw_star_field_image(  # type: ignore
        stars, resX, resY, star_intensity, star_sigma, integrated, lazy
    )  # type: ignore
