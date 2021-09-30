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

    Methods
    -------
    """

    def __init__(
        self, index: int, right_ascension: float, declination: float, magnitude: float
    ) -> None:
        self.index = index
        self.right_ascension = right_ascension
        self.declination = declination
        self.magnitude = magnitude

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self):
        return

    def __repr__(self) -> str:
        return f"Star({self.index}, {self.right_ascension}, {self.declination}, {self.magnitude})"

    def __str__(self) -> str:
        return f"""Star Class
        {self.index=},
        {self.right_ascension=},
        {self.declination=},
        {self.magnitude=},
        """
