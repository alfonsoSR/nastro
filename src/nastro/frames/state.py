from ..types.state import GenericState
from ..types.core import Vector, Double
from .reference_frames import Frame, ICRF
import numpy as np
from typing import Self


class CartesianPosition[T: (Double, Vector)](GenericState):
    """Three dimensional position vector in cartesian coordinates

    Components might be floating point numbers (single vector) or numpy arrays
    of them (time series of vectors).

    Parameters
    ----------
    x, y, z : Double | Vector
        Components of the position vector
    frame : Frame, optional
        Reference frame in which the position vector is defined

    Attributes
    ----------
    mag : Double | Vector
        Magnitude of the position vector
    """

    def __init__(self, x: T, y: T, z: T, *__args, frame: type[Frame] = ICRF) -> None:

        null = 0.0 * x
        super().__init__(x, y, z, null, null, null)
        self.frame = frame

        return None

    @property
    def x(self) -> T:
        return self.q1[0] if self.scalar else self.q1

    @property
    def y(self) -> T:
        return self.q2[0] if self.scalar else self.q2

    @property
    def z(self) -> T:
        return self.q3[0] if self.scalar else self.q3

    def asarray(self) -> Vector:
        return np.array([self.x, self.y, self.z])

    @property
    def mag(self) -> T:
        return np.linalg.norm(self.asarray(), axis=0)

    def __repr__(self) -> str:
        return (
            f"Position vector in {self.frame.__name__}: \n"
            "-------------------------------------------\n"
            f"x: {self.x}\n"
            f"y: {self.y}\n"
            f"z: {self.z}\n"
        )

    def convert(self, frame: type[Frame]) -> "CartesianPosition":
        """Convert position vector to a different reference frame

        Parameters
        ----------
        frame : Frame
            Reference frame in which the position vector is defined

        Returns
        -------
        CartesianPosition
            Position vector in the new reference frame
        """
        q1, q2, q3 = self.frame.transform(self.x, self.y, self.z, frame=frame)
        return CartesianPosition(q1, q2, q3, frame=frame)

    def to_spherical(self) -> "SphericalPosition":
        """Convert position vector to spherical coordinates

        Returns
        -------
        SphericalPosition
            Position vector in spherical coordinates
        """
        return SphericalPosition(
            *self.frame.to_spherical(self.x, self.y, self.z), frame=self.frame
        )


class CartesianVelocity[T: (Double, Vector)](GenericState):
    """Three dimensional velocity vector in cartesian coordinates

    Components might be floating point numbers (single vector) or numpy arrays
    of them (time series of vectors).

    Parameters
    ----------
    dx, dy, dz : Double | Vector
        Components of the velocity vector
    frame : Frame, optional
        Reference frame in which the velocity vector is defined

    Attributes
    ----------
    mag : Double | Vector
        Magnitude of the velocity vector
    """

    def __init__(self, dx: T, dy: T, dz: T, frame: type[Frame] = ICRF) -> None:

        null = 0.0 * dx
        super().__init__(null, null, null, dx, dy, dz)
        self.frame = frame

        return None

    @property
    def dx(self) -> T:
        return self.q4[0] if self.scalar else self.q4

    @property
    def dy(self) -> T:
        return self.q5[0] if self.scalar else self.q5

    @property
    def dz(self) -> T:
        return self.q6[0] if self.scalar else self.q6

    def asarray(self) -> Vector:
        return np.array([self.dx, self.dy, self.dz])

    @property
    def mag(self) -> T:
        return np.linalg.norm(self.asarray(), axis=0)

    def __repr__(self) -> str:
        return (
            f"Velocity vector in {self.frame.__name__}: \n"
            "-------------------------------------------\n"
            f"dx: {self.dx}\n"
            f"dy: {self.dy}\n"
            f"dz: {self.dz}\n"
        )


class SphericalPosition[T: (Double, Vector)](GenericState):
    """Three dimensional position vector in spherical coordinates

    Components might be floating point numbers (single vector) or numpy arrays
    of them (time series of vectors).

    Angles can be passed in radians (default) or degrees (deg=True), but they
    are always converted to radians during initialization. For each angle, the
    method SphericalPosition.angle will return the value in radians, while
    SphericalPosition.angle_deg will return it in degrees.
    """

    def __init__(
        self,
        r: T,
        theta: T,
        phi: T,
        frame: type[Frame] = ICRF,
        deg: bool = False,
        wrap: bool = False,
    ) -> None:

        null = 0.0 * r
        super().__init__(r, theta, phi, null, null, null)
        self.frame = frame

        if deg:
            self.q2 = np.deg2rad(self.q2)
            self.q3 = np.deg2rad(self.q3)
        if wrap:
            raise NotImplementedError(
                "Angle wrapping not implemented for spherical coordinates"
            )

        return None

    def to_cartesian(self) -> "CartesianPosition":
        """Convert position vector to cartesian coordinates

        Returns
        -------
        CartesianPosition
            Position vector in cartesian coordinates
        """
        return CartesianPosition(
            *self.frame.to_cartesian(self.q1, self.q2, self.q3), frame=self.frame
        )

    def __getattr__(self, name: str) -> T:

        if name in self.frame.variables:
            qi = self.__dict__[self.frame.variables[name]]
            return qi[0] if self.scalar else qi

        raise AttributeError(
            f"Attribute {name} is not defined for spherical position vector in"
            f" {self.frame.__name__}"
        )
