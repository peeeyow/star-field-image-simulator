import math
import numpy as np
import numpy.typing as npt
import pathlib
import sqlite3

from .constants import (
    ALPHA_MAX,
    ALPHA_MIN,
    DELTA_MAX,
    DELTA_MIN,
    HALF_REVOLUTION,
    REL,
)
from typing import Optional, Union


class Star:
    """
    Star class used to represent a star in the celestial coordinate system

    Attributes
    ----------
    index : int
        Star index, can be found the star catalog
    right_ascension : float
        Star right ascension location
        Represented in degrees
    declination : float
        Star declination location
        Represented in degrees
    magnitude : float
        star magnitude
    x : float
        Star x-coordinate location when celestial
            coordinates are converted to cartesian coordinates
    y : float
        Star y-coordinate location when celestial
            coordinates are converted to cartesian coordinates
    z : float
        Star z-coordinate location when celestial
            coordinates are converted to cartesian coordinates

    Methods
    -------
    compute_pixel_coordinate(camera_matrix)
        Computes for the u and v pixel-coordinates based
            from the given camera_matrix
    """

    def __init__(
        self,
        index: int,
        right_ascension: float,
        declination: float,
        magnitude: float,
    ) -> None:
        self.index = index
        self.right_ascension = right_ascension
        self.declination = declination
        self.magnitude = magnitude
        self.u = None
        self.v = None

    @property
    def X(self) -> float:
        """Returns X-coordinate of the star"""
        return math.cos(math.radians(self.declination)) * math.cos(
            math.radians(self.right_ascension)
        )

    @property
    def Y(self) -> float:
        """Returns Y-coordinate of the star"""
        return math.cos(math.radians(self.declination)) * math.sin(
            math.radians(self.right_ascension)
        )

    @property
    def Z(self) -> float:
        """Returns Z-coordinate of the star"""
        return math.sin(math.radians(self.declination))

    @property
    def u(self) -> Optional[float]:
        """Returns the u-pixel coordinate"""
        return self._u

    @u.setter
    def u(self, u: Optional[float]) -> None:
        """Sets the u-pixel coordinate"""
        self._u = u

    @property
    def v(self) -> Optional[float]:
        """Returns the v-pixel coordinate"""
        return self._v

    @v.setter
    def v(self, v: Optional[float]) -> None:
        """Sets the v-pixel coordinate"""
        self._v = v

    def compute_pixel_coordinate(self, camera_matrix: npt.ArrayLike) -> None:
        """Computes for the u and v pixel coordinates"""
        direction_vector = np.array([[self.X], [self.Y], [self.Z]])
        homogenous_vector = np.dot(camera_matrix, direction_vector).flatten()
        self.u, self.v, _ = homogenous_vector / homogenous_vector[-1]

    def __eq__(self, o: object) -> bool:
        """Check equality through star id, right ascension, declination,
        and magnitude"""
        if not isinstance(o, Star):
            return False

        if (
            self.index == o.index
            and math.isclose(
                self.right_ascension, o.right_ascension, abs_tol=REL
            )
            and math.isclose(self.declination, o.declination, abs_tol=REL)
            and math.isclose(self.magnitude, o.magnitude, abs_tol=REL)
        ):
            return True

        return False

    def __repr__(self) -> str:
        return f"Star({self.index}, {self.right_ascension}, \
            {self.declination}, {self.magnitude})"

    def __str__(self) -> str:
        return f"""
        Star Class
        {self.index=},
        {self.right_ascension=},
        {self.declination=},
        {self.magnitude=},
        {self.X=},
        {self.Y=},
        {self.Z=},
        """


class Celestial2Image:
    """
    Celestial2Image Class used to represent camera matrices and sub-matrices

    Attributes
    ----------
    alpha0 : float
        Camera boresight right_ascension
    delta0 : float
        Camera boresight declination
    phi0 : float
        Camera boresight roll
    fovX : float
        Camera horizontal field of view
    fovY : float
        Camera vertical field of view
    resX : int
        Camera horizontal pixel count (resolution)
    resY : int
        Camera vertical pixel count (resolution)
    rotation_matrix : numpy.ndarray[shape=(3,3), dtype[numpy.float]]
        Camera's external parameters
        Rotation matrix from celestial coordinate to sensor coordinate
    projection_matrix : numpy.ndarray[shape=(3,3), dtype[numpy.float]]
        Camera's internal parameters
        Projection matrix from sensor coordinate to image plane
    camara_matrix : numpy.ndarray[shape=(3,3), dtype[numpy.float]]
        Camera's complete parameters
        Matrix product of projection_matrix and rotation_matrix
    """

    def __init__(
        self,
        alpha0: float,
        delta0: float,
        phi0: float,
        fovX: float,
        fovY: float,
        resX: int,
        resY: int,
    ) -> None:
        self.alpha0 = alpha0
        self.delta0 = delta0
        self.phi0 = phi0
        self.fovX = fovX
        self.fovY = fovY
        self.resX = resX
        self.resY = resY

    @property
    def rotation_matrix(self) -> npt.ArrayLike:
        """Returns the rotation matrix."""
        a1 = math.sin(math.radians(self.alpha0)) * math.cos(
            math.radians(self.phi0)
        ) - math.cos(math.radians(self.alpha0)) * math.sin(
            math.radians(self.delta0)
        ) * math.sin(
            math.radians(self.phi0)
        )
        a2 = -math.sin(math.radians(self.alpha0)) * math.sin(
            math.radians(self.phi0)
        ) - math.cos(math.radians(self.alpha0)) * math.sin(
            math.radians(self.delta0)
        ) * math.cos(
            math.radians(self.phi0)
        )
        a3 = -math.cos(math.radians(self.alpha0)) * math.cos(
            math.radians(self.delta0)
        )
        b1 = -math.cos(math.radians(self.alpha0)) * math.cos(
            math.radians(self.phi0)
        ) - math.sin(math.radians(self.alpha0)) * math.sin(
            math.radians(self.delta0)
        ) * math.sin(
            math.radians(self.phi0)
        )
        b2 = math.cos(math.radians(self.alpha0)) * math.sin(
            math.radians(self.phi0)
        ) - math.sin(math.radians(self.alpha0)) * math.sin(
            math.radians(self.delta0)
        ) * math.cos(
            math.radians(self.phi0)
        )
        b3 = -math.sin(math.radians(self.alpha0)) * math.cos(
            math.radians(self.delta0)
        )
        c1 = math.cos(math.radians(self.delta0)) * math.sin(
            math.radians(self.phi0)
        )
        c2 = math.cos(math.radians(self.delta0)) * math.cos(
            math.radians(self.phi0)
        )
        c3 = -math.sin(math.radians(self.delta0))

        return np.transpose(
            np.array([[a1, a2, a3], [b1, b2, b3], [c1, c2, c3]])
        )

    @property
    def projection_matrix(self) -> npt.ArrayLike:
        """Returns projection matrix"""
        u0 = self.resX / 2
        v0 = self.resY / 2
        # u and v coordinate scaling
        fu = self.resX / (2 * math.tan(math.radians(self.fovX / 2)))
        fv = self.resY / (2 * math.tan(math.radians(self.fovY / 2)))
        # Intrinsic camera matrix
        return np.array([[fu, 0, u0], [0, -fv, v0], [0, 0, 1]])

    @property
    def camera_matrix(self):
        """Returns camera matrix"""
        return np.dot(
            self.projection_matrix,
            self.rotation_matrix,
        )

    def __repr__(self) -> str:
        return f"Celestial2Image( {self.alpha0}, {self.delta0}, {self.phi0},\
        {self.fovX}, {self.fovY}, {self.resX}, {self.resY},)"

    def __str__(self) -> str:
        return f"""
        Camera Properties
        Boresight:
            Right ascension: {self.alpha0},
            Declination: {self.delta0},
            Roll: {self.phi0},
        Field of View:
            Horizontal field of view: {self.fovX=},
            Vertical field of view: {self.fovY=},
        Image Resolution
            Horizontal resolution: {self.resX=},
            Vertical resolution: {self.resY=},
        """


"""
SQL wrapper functions
"""


def fetch_star_delta_is_northpole(
    curs: sqlite3.Cursor, radius: float, magnitude: float
) -> npt.ArrayLike:
    curs.execute(
        """SELECT * FROM star_catalog
        WHERE declination > :declination AND magnitude <= :magnitude;""",
        {"declination": DELTA_MAX - radius, "magnitude": magnitude},
    )
    return np.array(curs.fetchall())


def fetch_star_delta_is_southpole(
    curs: sqlite3.Cursor, radius: float, magnitude: float
) -> npt.ArrayLike:
    curs.execute(
        """SELECT * FROM star_catalog
        WHERE declination < :declination AND magnitude <= :magnitude;""",
        {"declination": DELTA_MIN + radius, "magnitude": magnitude},
    )
    return np.array(curs.fetchall())


def fetch_star_with_loop(
    curs: sqlite3.Cursor,
    dec_fov_min: float,
    dec_fov_max: float,
    magnitude: float,
) -> npt.ArrayLike:
    curs.execute(
        """SELECT * FROM star_catalog
        WHERE declination BETWEEN :dec_fov_min AND :dec_fov_max
        AND magnitude <= :magnitude;""",
        {
            "dec_fov_min": dec_fov_min,
            "dec_fov_max": dec_fov_max,
            "magnitude": magnitude,
        },
    )
    return np.array(curs.fetchall())


def fetch_star_with_overflow(
    curs: sqlite3.Cursor,
    ra_fov_min: float,
    ra_fov_max: float,
    dec_fov_min: float,
    dec_fov_max: float,
    magnitude: float,
) -> npt.ArrayLike:
    curs.execute(
        """SELECT * FROM star_catalog
        WHERE right_ascension NOT BETWEEN :ra_fov_max AND :ra_fov_min
        AND  declination BETWEEN :dec_fov_min AND :dec_fov_max
        AND magnitude <= :magnitude;""",
        {
            "ra_fov_min": ra_fov_min,
            "ra_fov_max": ra_fov_max,
            "dec_fov_min": dec_fov_min,
            "dec_fov_max": dec_fov_max,
            "magnitude": magnitude,
        },
    )
    return np.array(curs.fetchall())


def fetch_star_no_loop(
    curs: sqlite3.Cursor,
    ra_fov_min: float,
    ra_fov_max: float,
    dec_fov_min: float,
    dec_fov_max: float,
    magnitude: float,
) -> npt.ArrayLike:
    curs.execute(
        """SELECT * FROM star_catalog
        WHERE right_ascension BETWEEN :ra_fov_min AND :ra_fov_max
        AND  declination BETWEEN :dec_fov_min AND :dec_fov_max
        AND magnitude <= :magnitude;""",
        {
            "ra_fov_min": ra_fov_min,
            "ra_fov_max": ra_fov_max,
            "dec_fov_min": dec_fov_min,
            "dec_fov_max": dec_fov_max,
            "magnitude": magnitude,
        },
    )
    return curs.fetchall()


def fetch_stars(
    alpha0: float,
    delta0: float,
    fovX: float,
    fovY: float,
    magnitude: float,
    path: Union[pathlib.Path, str],
) -> npt.ArrayLike:
    conn = sqlite3.connect(path)
    curs = conn.cursor()

    radius = np.sqrt(fovX ** 2 + fovY ** 2) / 2

    if delta0 == 90:
        return fetch_star_delta_is_northpole(curs, radius, magnitude)

    if delta0 == -90:
        return fetch_star_delta_is_southpole(curs, radius, magnitude)

    dec_fov_min = max(delta0 - radius, DELTA_MIN)
    dec_fov_max = min(delta0 + radius, DELTA_MAX)

    if radius / math.cos(math.radians(delta0)) >= HALF_REVOLUTION:
        return fetch_star_with_loop(curs, dec_fov_min, dec_fov_max, magnitude)

    ra_fov_min = alpha0 - radius / np.cos(np.radians(delta0))
    ra_fov_max = alpha0 + radius / np.cos(np.radians(delta0))

    if ra_fov_max >= ALPHA_MAX or ra_fov_min <= ALPHA_MIN:
        ra_fov_min %= ALPHA_MAX
        ra_fov_max %= ALPHA_MAX
        return fetch_star_with_overflow(
            curs, ra_fov_min, ra_fov_max, dec_fov_min, dec_fov_max, magnitude
        )

    return fetch_star_no_loop(
        curs, ra_fov_min, ra_fov_max, dec_fov_min, dec_fov_max, magnitude
    )


def create_stars(
    alpha0: float,
    delta0: float,
    fovX: float,
    fovY: float,
    magnitude: float,
    path: Union[pathlib.Path, str],
) -> list[Star]:
    stars_as_list = fetch_stars(alpha0, delta0, fovX, fovY, magnitude, path)
    return [  # type: ignore
        Star(idx, ra, dec, mag)  # type: ignore
        for idx, ra, dec, mag in stars_as_list  # type: ignore
    ]  # type: ignore


# def simulate_sfim(
#     alpha0: float,
#     delta0: float,
#     phi0: float,
#     fovX: float,
#     fovY: float,
#     magnitude: float,
#     path: Union[pathlib.Path, str],
# ) -> list[Star]:
