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
    create_stars,
    remove_random_stars,
)


def generate_star_field_image(
    stars: list[Star],
    resX: int,
    resY: int,
    star_intensity: float,
    star_sigma: float,
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
        # input to erf
        x = indices_u[scope] - Ui
        y = indices_v[scope] - Vi

        # computing starContribution
        starContribution = (star_intensity / STAR_INTENSITY_LEVEL ** Mi) * (
            np.exp(-(x ** 2 + y ** 2) / (2 * star_sigma ** 2))
        )

        star_field_image[scope] += starContribution
    return star_field_image, centroids


def simulate_star_field_image(
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
    lazy: bool,
) -> npt.ArrayLike:
    c2i = Celestial2Image(alpha0, delta0, phi0, fovX, fovY, resX, resY)
    stars = create_stars(
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
    return generate_star_field_image(  # type: ignore
        stars, resX, resY, star_intensity, star_sigma, lazy  # type: ignore
    )  # type: ignore


def generate_centroids(
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
):
    c2i = Celestial2Image(alpha0, delta0, phi0, fovX, fovY, resX, resY)
    stars = create_stars(
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

    return [(star.u, star.v) for star in stars]
