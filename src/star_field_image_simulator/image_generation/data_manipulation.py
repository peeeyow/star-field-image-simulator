import math
import numpy


class Star:
    """
    Star class use to represent a star in the celestial coordinate system

    Attributes
    ----------
    index : int
        Star index, can be found the star catalog.
    right_ascension : float
        Star right ascension location.
        Represented in degrees.
    declination : float
        Star declination location.
        Represented in degrees.
    magnitude : float
        star magnitude
    x : float
        Star x-coordinate location when celestial coordinates are converted to cartesian coordinates
    y : float
        Star y-coordinate location when celestial coordinates are converted to cartesian coordinates
    z : float
        Star z-coordinate location when celestial coordinates are converted to cartesian coordinates
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

    @property
    def X(self):
        """Returns X-coordinate of the star"""
        return math.cos(math.radians(self.declination)) * math.cos(
            math.radians(self.right_ascension)
        )

    @property
    def Y(self):
        """Returns Y-coordinate of the star"""
        return math.cos(math.radians(self.declination)) * math.sin(
            math.radians(self.right_ascension)
        )

    @property
    def Z(self):
        """Returns Z-coordinate of the star"""
        return math.sin(math.radians(self.declination))

    def __repr__(self) -> str:
        return f"Star({self.index}, {self.right_ascension}, {self.declination}, {self.magnitude})"

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
