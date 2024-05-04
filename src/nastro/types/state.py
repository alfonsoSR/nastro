import numpy as np
from typing import Self, Any, Iterator
from pathlib import Path
from .core import Double, Vector, Scalar, is_double, is_vector, is_scalar
import traceback
from ..constants import halfpi, pi, twopi


class GenericState[U: (Double, Vector)]:
    """Base class for state vectors

    Implements operations and basic functionality that is common to all types
    of state vectors and implements the containers that are used internally
    to manipulate their components. The elements q1 to q6 are representations
    of the six components of a state vector as numpy arrays.

    This class is not meant to be instantiated directly, but to be subclassed
    to create specific types of state vectors: cartesian, keplerian, etc.
    """

    __slots__ = ("q1", "q2", "q3", "q4", "q5", "q6")
    properties = {
        "q1": "q1",
        "q2": "q2",
        "q3": "q3",
        "q4": "q4",
        "q5": "q5",
        "q6": "q6",
    }
    angles: dict[str, tuple[Double, Double]] = {}

    def __init__(
        self,
        q1: U,
        q2: U,
        q3: U,
        q4: U,
        q5: U,
        q6: U,
        deg: bool = False,
        wrap: bool = False,
    ) -> None:

        if is_double(q1):
            _input = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float64)[:, None]
            assert len(_input.shape) == 2
            assert _input.shape == (6, 1)
        elif is_vector(q1):
            try:
                _input = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float64)
                assert len(_input.shape) == 2
                assert _input.shape[0] == 6
            except ValueError or AssertionError:
                raise ValueError(
                    f"Failed to create {self.__class__.__name__}. "
                    "Components have different sizes."
                )
        else:
            raise TypeError(
                f"Failed to create {self.__class__.__name__}. "
                "Components must be floating point numbers or vectors."
            )

        # Initialize generic state components
        self.q1 = np.array(q1, dtype=np.float64, ndmin=1)
        self.q2 = np.array(q2, dtype=np.float64, ndmin=1)
        self.q3 = np.array(q3, dtype=np.float64, ndmin=1)
        self.q4 = np.array(q4, dtype=np.float64, ndmin=1)
        self.q5 = np.array(q5, dtype=np.float64, ndmin=1)
        self.q6 = np.array(q6, dtype=np.float64, ndmin=1)

        if len(self.angles.keys()) == 0:
            return None

        if deg:
            for key in self.angles.keys():
                setattr(
                    self,
                    self.properties[key],
                    np.deg2rad(getattr(self, self.properties[key])),
                )
        if wrap:
            for key, limits in self.angles.items():
                setattr(
                    self,
                    self.properties[key],
                    self.__wrap_angle(getattr(self, self.properties[key]), limits),
                )

        return None

    @property
    def size(self) -> int:
        """Number of state vectors in time series"""
        return self.q1.size

    @property
    def scalar(self) -> bool:
        """True for a single state vector"""
        return self.q1.size == 1

    @property
    def asarray(self) -> Vector:
        """Time series of state vectors as a (6, N) numpy array"""
        return np.array(
            [self.q1, self.q2, self.q3, self.q4, self.q5, self.q6],
            dtype=np.float64,
        )

    @staticmethod
    def __wrap_angle(angle: U, limits: tuple[Double, Double] = (0.0, twopi)) -> U:
        """Wrap angle to limits"""

        low, high = limits
        angle %= twopi
        if low == 0 and high == twopi:
            return angle

        if low == 0 and high == pi:
            return angle % pi

        out = angle - ((angle + pi) // twopi) * twopi
        if low == -pi and high == pi:
            return out

        positive = (out * (out < halfpi) + (pi - out) * (out >= halfpi)) * (out >= 0)
        negative = (out * (out > -halfpi) - (pi + out) * (out <= -halfpi)) * (out < 0)

        if low == -halfpi and high == halfpi:
            return positive + negative

        raise ValueError("Failed to wrap angle: Invalid limits.")

    def __getattr__(self, name: str) -> U:

        if name in self.properties:
            qi = getattr(self, self.properties[name])
            return qi[0] if self.scalar else qi

        if name[-4:] == "_deg" and name[:-4] in self.angles:
            key = name[:-4]
            return np.rad2deg(getattr(self, key))

        raise AttributeError(f"{name} is not a property of {self.__class__.__name__}! ")

    def __getitem__(self, index: int | slice) -> Self:

        return type(self)(*self.asarray[:, index])

    def __iter__(self) -> Iterator[Self]:
        return iter([self.__getitem__(idx) for idx in range(self.size)])

    def __repr__(self) -> str:

        out = f"{self.__class__.__name__}\n"
        out += "-" * len(out) + "\n"
        for key in self.properties:
            out += f"{key}: {getattr(self, key)}\n"
        return out

    def __add__(self, other: Self | Scalar) -> Self:

        if is_scalar(other):
            return type(self)(*(self.asarray + other))
        elif isinstance(other, self.__class__):
            return type(self)(*(self.asarray + other.asarray))
        else:
            raise TypeError(
                "Failed to perform addition! Check documentation for supported types."
            )

    def __sub__(self, other: Self | Scalar) -> Self:

        if is_scalar(other):
            return type(self)(*(self.asarray - other))
        elif isinstance(other, self.__class__):
            return type(self)(*(self.asarray - other.asarray))
        else:
            raise TypeError(
                "Failed to perform subtraction! "
                "Check documentation for supported types."
            )

    def __mul__(self, other: Self | Scalar) -> Self:

        if is_scalar(other):
            return type(self)(*(self.asarray * other))
        elif isinstance(other, self.__class__):
            return type(self)(*(self.asarray * other.asarray))
        else:
            raise TypeError(
                "Failed to perform multiplication! "
                "Check documentation for supported types."
            )

    def __truediv__(self, other: Self | Scalar) -> Self:

        if is_scalar(other):
            return type(self)(*(self.asarray / other))
        elif isinstance(other, self.__class__):
            return type(self)(*(self.asarray / other.asarray))
        else:
            raise TypeError(
                "Failed to perform division! "
                "Check documentation for supported types."
            )

    def __eq__(self, other: object) -> bool:

        if not isinstance(other, self.__class__):
            raise TypeError("Comparison is only supported for state vectors")

        return np.allclose(self.asarray, other.asarray, atol=1e-15, rtol=0.0)

    def append(self, other: Self | Scalar | Vector) -> Self:
        """Add state vector to time series"""

        if is_scalar(other):
            return type(self)(*np.append(self.asarray, np.ones((6, 1)) * other, axis=1))
        elif is_vector(other):
            if other.shape[0] != 6:
                raise ValueError(
                    "Failed to append to state vector. Input must be a (6, N) array."
                )
            if len(other.shape) == 1:
                other = other[:, None]
            return type(self)(*np.append(self.asarray, other, axis=1))
        elif isinstance(other, self.__class__):
            return type(self)(*np.append(self.asarray, other.asarray, axis=1))
        else:
            raise TypeError(
                "Failed to append to state vector. "
                "Check documentation for supported types."
            )

    def copy(self) -> Self:
        """Return copy of state vector"""
        return type(self)(*self.asarray)

    # IO
    def save(self, path: str | Path, relative: bool = True) -> Path:

        # Add extension if missing
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npy")

        # Generate absolute path
        cwd = Path("/")
        if relative:
            cwd = Path(traceback.extract_stack()[-2].filename).parent
        path = cwd / path

        # Save output
        np.save(path, self.asarray)

        return path

    @classmethod
    def load(cls, path: str | Path, relative: bool = True) -> Self:

        # Add extension if missing
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npy")

        # Generate absolute path
        cwd = Path("/")
        if relative:
            cwd = Path(traceback.extract_stack()[-2].filename).parent
        path = cwd / path

        # Load output
        return cls(*np.load(path))

    @classmethod
    def from_tudat(cls, state_history: dict[str, Vector]) -> Self:
        raise NotImplementedError("Missing conversion from TUDAT state history")

    # Frame conversions
    def transform(self) -> Self:
        """Transform state to a different reference frame"""
        raise NotImplementedError("Frame transformations are not implemented yet")


class CartesianState[U: (Double, Vector)](GenericState[U]):
    """Cartesian state vector

    Single state or a time series of states of a body in a three-dimensional
    space in terms of cartesian coordinates. Each component of the state vector
    can be a floating point number (single state) or a numpy array of them (time
    series of states).

    :param x: X component of the position vector [m]
    :param y: Y component of the position vector [m]
    :param z: Z component of the position vector [m]
    :param dx: X component of the velocity vector [m/s]
    :param dy: Y component of the velocity vector [m/s]
    :param dz: Z component of the velocity vector [m/s]
    """

    properties = {
        "x": "q1",
        "y": "q2",
        "z": "q3",
        "dx": "q4",
        "dy": "q5",
        "dz": "q6",
    }

    def __init__(self, x: U, y: U, z: U, dx: U, dy: U, dz: U) -> None:
        super().__init__(x, y, z, dx, dy, dz)

    @property
    def r_vec(self) -> Vector:
        """Cartesian position vector as numpy array"""
        return self.asarray[:3]

    @property
    def r_mag(self) -> U:
        """Magnitude of the position vector"""
        return np.linalg.norm(self.r_vec, axis=0)

    @property
    def v_vec(self) -> Vector:
        """Cartesian velocity vector as numpy array"""
        return self.asarray[3:]

    @property
    def v_mag(self) -> U:
        """Magnitude of the velocity vector"""
        return np.linalg.norm(self.v_vec, axis=0)

    def to_keplerian(self, mu: Double) -> "KeplerianState":
        """Conversion to keplerian state vector

        Algorithm: Practical astrodynamics slides

        :param mu: Gravitational parameter of central body [m^3/s^2]
        """
        # TODO: Ensure that loss of accuracy in np.sum is not relevant
        # TODO: Only allow conversion if frame is valid

        # Semi-major axis
        a = 1.0 / ((2.0 / self.r_mag) - (self.v_mag * self.v_mag / mu))

        # Angular momentum
        h_vec = np.cross(self.r_vec, self.v_vec, axis=0)
        h = np.linalg.norm(h_vec, axis=0)

        # Eccentricity
        e_vec = (np.cross(self.v_vec, h_vec, axis=0) / mu) - (self.r_vec / self.r_mag)
        e = np.linalg.norm(e_vec, axis=0)
        e_uvec = e_vec / e

        # Inclination
        inc = np.arccos(h_vec[2] / h)

        # N vector
        _z_vec = np.array(
            [np.zeros_like(self.x), np.zeros_like(self.y), np.ones_like(self.z)]
        )
        N_vec = np.cross(_z_vec, h_vec, axis=0)
        Nxy = np.sqrt(N_vec[0] * N_vec[0] + N_vec[1] * N_vec[1])
        N_uvec = N_vec / Nxy

        # RAAN
        raan = np.arctan2(N_vec[1] / Nxy, N_vec[0] / Nxy)

        # Argument of periapsis
        sign_aop_condition = np.sum(np.cross(N_uvec, e_vec, axis=0) * h_vec, axis=0) > 0
        sign_aop = 2 * sign_aop_condition - 1
        aop = sign_aop * np.arccos(np.sum(e_uvec * N_uvec, axis=0))

        # True anomaly
        sign_ta_condition = (
            np.sum(np.cross(e_vec, self.r_vec, axis=0) * h_vec, axis=0) > 0
        )
        sign_ta = 2 * sign_ta_condition - 1
        ta = sign_ta * np.arccos(np.sum(self.r_vec * e_uvec / self.r_mag, axis=0))

        return KeplerianState(a, e, inc, raan, aop, ta, deg=False)

    def to_spherical(self):
        raise NotImplementedError(
            "Conversion to spherical state is not implemented yet."
        )


class CartesianStateDerivative[U: (Double, Vector)](GenericState[U]):
    """Derivative of cartesian state

    Derivative of a single state of a time series of states of a body in a
    three-dimensional space in terms of cartesian coordinates. Each component
    can be a floating point number (single state) or a numpy array of them
    (time series of states).

    :param dx: X component of the velocity vector [m/s]
    :param dy: Y component of the velocity vector [m/s]
    :param dz: Z component of the velocity vector [m/s]
    :param ddx: X component of the acceleration vector [m/s^2]
    :param ddy: Y component of the acceleration vector [m/s^2]
    :param ddz: Z component of the acceleration vector [m/s^2]
    """

    properties = {
        "dx": "q1",
        "dy": "q2",
        "dz": "q3",
        "ddx": "q4",
        "ddy": "q5",
        "ddz": "q6",
    }

    def __init__(self, dx: U, dy: U, dz: U, ddx: U, ddy: U, ddz: U) -> None:
        super().__init__(dx, dy, dz, ddx, ddy, ddz)
        return None

    @property
    def v_vec(self) -> Vector:
        """Cartesian velocity vector as numpy array"""
        return self.asarray[:3]

    @property
    def a_vec(self) -> Vector:
        """Cartesian acceleration vector as numpy array"""
        return self.asarray[3:]

    @property
    def v_mag(self) -> U:
        """Magnitude of the velocity vector"""
        return np.linalg.norm(self.v_vec, axis=0)

    @property
    def a_mag(self) -> U:
        """Magnitude of the acceleration vector"""
        return np.linalg.norm(self.a_vec, axis=0)

    def times_dt(self, dt: Double) -> CartesianState:
        """Change in cartesian state over interval dt

        Returns the change of a cartesian state with this object as derivative
        over a time interval dt. The main use case are numerical integrators,
        in which the state is updated by multiplying its derivative by the
        time step.

        :param dt: Time step
        :return: Change in cartesian elements over interval dt
        """

        return CartesianState(*(self.asarray * dt))

    def add_acceleration(self, other: Self) -> None:
        """Update the acceleration of the cartesian state derivative

        Update the acceleration of this object by adding the components of the
        acceleration vector of other cartesian state derivative. The use case is
        to add up all the accelerations acting on a body during orbit propagation.
        Since each acceleration is defined by a cartesian state derivative,
        adding them up directly would also update the velocity, which is incorrect.

        :param other: Cartesian state derivative with acceleration to add
        """

        if not other.scalar:
            raise NotImplementedError(
                "Failed to add acceleration. Vector input not supported."
            )

        # q4, q5 and q6 are the components of a general state vector that
        # represent the acceleration in CartesianStateDerivative
        self.q4 += other.q4
        self.q5 += other.q5
        self.q6 += other.q6

        return None


class KeplerianState[U: (Double, Vector)](GenericState[U]):
    """Keplerian state vector

    Single state of a time series of states of a body in a three-dimensional
    space in terms of classical keplerian elements. Each component of the state
    vector can be a floating point number (single state) or a numpy array of them
    (time series of states).

    Angles might be given in degrees or radians, but are always converted to
    radians during initialization. For each angle, the method KeplerianState.angle
    will return the value in radians, while KeplerianState.angle_deg will return
    it in degrees.

    Angles can be wrapped if specified by the wrap parameter, which is activated
    by default. Wrapping will limit inclinations to the range [0, pi) and the
    rest of the angles to [0, 2pi).

    :param a: Semi-major axis [m]
    :param e: Eccentricity
    :param i: Inclination [rad]
    :param raan: Right ascension of the ascending node [rad]
    :param aop: Argument of periapsis [rad]
    :param ta: True anomaly [rad]
    :param deg: Angles are given in degrees (True) or radians (False)
    :param wrap: Wrap angles (True) or not (False)
    """

    properties = {
        "a": "q1",
        "e": "q2",
        "i": "q3",
        "raan": "q4",
        "aop": "q5",
        "ta": "q6",
    }

    angles = {
        "i": (0.0, pi),
        "raan": (0.0, twopi),
        "aop": (0.0, twopi),
        "ta": (0.0, twopi),
    }

    def __init__(
        self,
        a: U,
        e: U,
        i: U,
        raan: U,
        aop: U,
        ta: U,
        deg: bool = False,
        wrap: bool = True,
    ) -> None:
        super().__init__(a, e, i, raan, aop, ta, deg=deg, wrap=wrap)
        return None

    @property
    def E(self) -> U:
        """Eccentric anomaly"""
        if np.any(self.e >= 1.0) or np.any(self.e < 0.0):
            raise NotImplementedError(
                "Failed to compute eccentric anomaly. "
                "Only circular and elliptical orbits are supported"
            )
        out: Any = 2.0 * np.arctan2(
            np.sqrt(1.0 - self.e) * np.sin(0.5 * self.ta),
            np.sqrt(1.0 + self.e) * np.cos(0.5 * self.ta),
        )
        return out

    @property
    def M(self) -> U:
        """Mean anomaly"""
        out: Any = self.E - self.e * np.sin(self.E)
        return out

    @property
    def T(self) -> U:
        """Orbital period"""
        out: Any = 2.0 * np.pi * np.sqrt(self.a**3)
        return out

    def __sub__(self, other: Self | Scalar) -> Self:
        print(
            "WARNING: Keplerian state subtraction is not properly tested\n"
            "It might produce unexpected results"
        )

        if not isinstance(other, self.__class__):
            raise NotImplementedError(
                "Subtraction from KeplerianState is only supported "
                "for KeplerianState objects."
            )

        da: Any = self.q1 - other.q1
        de: Any = self.q2 - other.q2
        di: Any = np.unwrap(self.q3, period=pi) - np.unwrap(other.q3, period=pi)
        draan: Any = np.unwrap(self.q4, period=twopi) - np.unwrap(
            other.q4, period=twopi
        )
        daop: Any = np.unwrap(self.q5, period=twopi) - np.unwrap(other.q5, period=twopi)
        dta: Any = np.unwrap(self.q6, period=twopi) - np.unwrap(other.q6, period=twopi)

        return type(self)(da, de, di, draan, daop, dta, wrap=False)

    def to_cartesian(self, mu: Double) -> "CartesianState":
        """Conversion to cartesian state vector

        Algorithm: Practical astrodynamics slides

        :param mu: Gravitational parameter of central body [m^3/s^2]
        """

        # Auxiliary trigonometric relations
        cos_Omega = np.cos(self.raan)
        sin_Omega = np.sin(self.raan)
        cos_omega = np.cos(self.aop)
        sin_omega = np.sin(self.aop)
        cos_i = np.cos(self.i)
        sin_i = np.sin(self.i)
        cos_theta = np.cos(self.ta)
        sin_theta = np.sin(self.ta)

        l1 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i
        l2 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i
        m1 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i
        m2 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i
        n1 = sin_omega * sin_i
        n2 = cos_omega * sin_i

        # Orbital radius
        r = self.a * (1.0 - self.e * self.e) / (1.0 + self.e * cos_theta)

        # Position in orbital plane
        xi = r * cos_theta
        eta = r * sin_theta

        # Position in 3D space
        x = l1 * xi + l2 * eta
        y = m1 * xi + m2 * eta
        z = n1 * xi + n2 * eta

        # Angular momentum
        H = np.sqrt(mu * self.a * (1.0 - self.e * self.e))

        # Velocity
        common = mu / H
        e_cos_theta = self.e + cos_theta

        dx = common * (l2 * e_cos_theta - l1 * sin_theta)
        dy = common * (m2 * e_cos_theta - m1 * sin_theta)
        dz = common * (n2 * e_cos_theta - n1 * sin_theta)

        return CartesianState(x, y, z, dx, dy, dz)


class KeplerianStateDerivative[U: (Double, Vector)](GenericState[U]):
    """Derivative of keplerian state

    Derivative of a single state of a time series of states of a body in a
    three-dimensional space in terms of classical keplerian elements. Each
    component can be a floating point number (single state) or a numpy array
    of them (time series of states).

    :param da: Rate of change of semi-major axis [m/s]
    :param de: Rate of change of eccentricity [1/s]
    :param di: Rate of change of inclination [rad/s]
    :param draan: Rate of change of right ascension of the ascending node [rad/s]
    :param daop: Rate of change of argument of periapsis [rad/s]
    :param dta: Rate of change of true anomaly [rad/s]
    """

    properties = {
        "da": "q1",
        "de": "q2",
        "di": "q3",
        "draan": "q4",
        "daop": "q5",
        "dta": "q6",
    }

    def __init__(self, da: U, de: U, di: U, draan: U, daop: U, dta: U) -> None:
        super().__init__(da, de, di, draan, daop, dta)
        return None

    def times_dt(self, dt: Double) -> KeplerianState:
        """Change in keplerian state over interval dt

        Returns the change of a keplerian state with this object as derivative
        over a time interval dt. The main use case are numerical integrators,
        in which the state is updated by multiplying its derivative by the
        time step.

        :param dt: Time step
        :return: Change in keplerian elements over interval dt
        """

        return KeplerianState(*(self.asarray * dt))


class SphericalState[U: (Double, Vector)](GenericState):

    def __init__(
        self,
        rho: U,
        theta: U,
        phi: U,
        drho: U,
        dtheta: U,
        dphi: U,
        deg: bool = False,
        wrap: bool = True,
    ) -> None:
        raise NotImplementedError("SphericalState is not implemented yet.")

    def __add__(self, other: object) -> Self:
        raise NotImplementedError(
            "Addition is still not supported for spherical state vectors"
        )

    def __sub__(self, other: object) -> Self:
        raise NotImplementedError(
            "Subtraction is still not supported for spherical state vectors"
        )

    def transform(self) -> Self:
        raise NotImplementedError(
            "Frame transformations are not implemented yet for spherical state vectors"
        )

    def to_cartesian(self) -> "CartesianState":
        """Conversion from spherical to cartesian state"""
        raise NotImplementedError(
            "Conversion from spherical to cartesian state is not implemented yet."
        )

    def to_keplerian(self, mu: Double) -> "KeplerianState":
        """Conversion from spherical to keplerian state"""
        raise NotImplementedError(
            "Conversion from spherical to keplerian state is not implemented yet."
        )


class CartesianPosition[U: (Double, Vector)](GenericState[U]):
    """Three dimensional position vector in cartesian coordinates

    Components might be floating point numbers (single vector) or numpy arrays
    of them (time series of vectors).

    :param x: X coordinate [m]
    :param y: Y coordinate [m]
    :param z: Z coordinate [m]
    """

    properties = {"x": "q1", "y": "q2", "z": "q3"}

    def __init__(self, x: U, y: U, z: U, *_) -> None:
        super().__init__(x, y, z, 0.0 * x, 0.0 * y, 0.0 * z)
        return None

    @property
    def asarray(self) -> Vector:
        """Time series of position vectors as a (3, N) numpy array"""
        return np.array([self.q1, self.q2, self.q3], dtype=np.float64)


class CartesianVelocity[U: (Double, Vector)](GenericState[U]):
    """Three dimensional velocity vector in cartesian coordinates

    Components might be floating point numbers (single vector) or numpy arrays
    of them (time series of vectors).

    :param dx: X component of the velocity vector [m/s]
    :param dy: Y component of the velocity vector [m/s]
    :param dz: Z component of the velocity vector [m/s]
    """

    properties = {"dx": "q4", "dy": "q5", "dz": "q6"}

    def __init__(self, dx: U, dy: U, dz: U, *_) -> None:
        super().__init__(0.0 * dx, 0.0 * dy, 0.0 * dz, dx, dy, dz)
        return None

    @property
    def asarray(self) -> Vector:
        """Time series of velocity vectors as a (3, N) numpy array"""
        return np.array([self.q4, self.q5, self.q6], dtype=np.float64)


class SphericalPosition[U: (Double, Vector)](GenericState[U]):

    def __init__(self) -> None:
        raise NotImplementedError("SphericalPosition is not implemented yet.")


# class _CartesianState[U: (Double, Vector)](GenericState):
#     """Cartesian state vector

#     Single state or a time series of states of a body in a three-dimensional
#     space in terms of cartesian coordinates. Each component of the state vector
#     can be a floating point number (single state) or a numpy array of them (time
#     series of states).

#     :param x: X component of the position vector [m]
#     :param y: Y component of the position vector [m]
#     :param z: Z component of the position vector [m]
#     :param dx: X component of the velocity vector [m/s]
#     :param dy: Y component of the velocity vector [m/s]
#     :param dz: Z component of the velocity vector [m/s]
#     """

#     def __init__(self, x: U, y: U, z: U, dx: U, dy: U, dz: U) -> None:
#         super().__init__(x, y, z, dx, dy, dz)
#         return None

#     @property
#     def x(self) -> U:
#         return self.q1[0] if self.scalar else self.q1

#     @property
#     def y(self) -> U:
#         return self.q2[0] if self.scalar else self.q2

#     @property
#     def z(self) -> U:
#         return self.q3[0] if self.scalar else self.q3

#     @property
#     def dx(self) -> U:
#         return self.q4[0] if self.scalar else self.q4

#     @property
#     def dy(self) -> U:
#         return self.q5[0] if self.scalar else self.q5

#     @property
#     def dz(self) -> U:
#         return self.q6[0] if self.scalar else self.q6

#     @property
#     def r_vec(self) -> Vector:
#         return np.array([self.x, self.y, self.z])

#     @property
#     def v_vec(self) -> Vector:
#         return np.array([self.dx, self.dy, self.dz])

#     @property
#     def r_mag(self) -> U:
#         return np.linalg.norm(self.r_vec, axis=0)

#     @property
#     def v_mag(self) -> U:
#         return np.linalg.norm(self.v_vec, axis=0)

#     def __repr__(self) -> str:
#         return (
#             "Cartesian state vector\n"
#             "----------------------\n"
#             f"x: {self.x}\n"
#             f"y: {self.y}\n"
#             f"z: {self.z}\n"
#             f"dx: {self.dx}\n"
#             f"dy: {self.dy}\n"
#             f"dz: {self.dz}"
#         )

#     def as_keplerian(self, mu: Double) -> "KeplerianState":
#         """Conversion to keplerian state vector

#         Algorithm: Practical astrodynamics slides

#         :param mu: Gravitational parameter of central body [m^3/s^2]
#         """
#         # TODO: Ensure that loss of accuracy in np.sum is not relevant

#         # Semi-major axis
#         a = 1.0 / ((2.0 / self.r_mag) - (self.v_mag * self.v_mag / mu))

#         # Angular momentum
#         h_vec = np.cross(self.r_vec, self.v_vec, axis=0)
#         h = np.linalg.norm(h_vec, axis=0)

#         # Eccentricity
#         e_vec = (np.cross(self.v_vec, h_vec, axis=0) / mu) - (self.r_vec / self.r_mag)
#         e = np.linalg.norm(e_vec, axis=0)
#         e_uvec = e_vec / e

#         # Inclination
#         inc = np.arccos(h_vec[2] / h)

#         # N vector
#         _z_vec = np.array(
#             [np.zeros_like(self.x), np.zeros_like(self.y), np.ones_like(self.z)]
#         )
#         N_vec = np.cross(_z_vec, h_vec, axis=0)
#         Nxy = np.sqrt(N_vec[0] * N_vec[0] + N_vec[1] * N_vec[1])
#         N_uvec = N_vec / Nxy

#         # RAAN
#         raan = np.arctan2(N_vec[1] / Nxy, N_vec[0] / Nxy)

#         # Argument of periapsis
#         sign_aop_condition = np.sum(np.cross(N_uvec, e_vec, axis=0) * h_vec, axis=0) > 0
#         sign_aop = 2 * sign_aop_condition - 1
#         aop = sign_aop * np.arccos(np.sum(e_uvec * N_uvec, axis=0))

#         # True anomaly
#         sign_ta_condition = (
#             np.sum(np.cross(e_vec, self.r_vec, axis=0) * h_vec, axis=0) > 0
#         )
#         sign_ta = 2 * sign_ta_condition - 1
#         ta = sign_ta * np.arccos(np.sum(self.r_vec * e_uvec / self.r_mag, axis=0))

#         return KeplerianState(a, e, inc, raan, aop, ta, deg=False)


# class _CartesianStateDerivative[U: (Double, Vector)](GenericState):
#     """Derivative of cartesian state

#     Derivative of a single state of a time series of states of a body in a
#     three-dimensional space in terms of cartesian coordinates. Each component
#     can be a floating point number (single state) or a numpy array of them
#     (time series of states).

#     :param dx: X component of the velocity vector [m/s]
#     :param dy: Y component of the velocity vector [m/s]
#     :param dz: Z component of the velocity vector [m/s]
#     :param ddx: X component of the acceleration vector [m/s^2]
#     :param ddy: Y component of the acceleration vector [m/s^2]
#     :param ddz: Z component of the acceleration vector [m/s^2]
#     """

#     def __init__(self, dx: U, dy: U, dz: U, ddx: U, ddy: U, ddz: U) -> None:
#         super().__init__(dx, dy, dz, ddx, ddy, ddz)
#         return None

#     @property
#     def dx(self) -> U:
#         return self.q1[0] if self.scalar else self.q1

#     @property
#     def dy(self) -> U:
#         return self.q2[0] if self.scalar else self.q2

#     @property
#     def dz(self) -> U:
#         return self.q3[0] if self.scalar else self.q3

#     @property
#     def ddx(self) -> U:
#         return self.q4[0] if self.scalar else self.q4

#     @property
#     def ddy(self) -> U:
#         return self.q5[0] if self.scalar else self.q5

#     @property
#     def ddz(self) -> U:
#         return self.q6[0] if self.scalar else self.q6

#     @property
#     def v_vec(self) -> Vector:
#         return np.array([self.dx, self.dy, self.dz])

#     @property
#     def a_vec(self) -> Vector:
#         return np.array([self.ddx, self.ddy, self.ddz])

#     @property
#     def v_mag(self) -> U:
#         return np.linalg.norm(self.v_vec, axis=0)

#     @property
#     def dv_mag(self) -> U:
#         return np.linalg.norm(self.v_vec, axis=0)

#     def times_dt(self, dt: Double) -> CartesianState:
#         """Change in cartesian state over interval dt

#         Returns the change of a cartesian state with this object as derivative
#         over a time interval dt. The main use case are numerical integrators,
#         in which the state is updated by multiplying its derivative by the
#         time step.

#         :param dt: Time step
#         :return: Change in cartesian elements over interval dt
#         """

#         return CartesianState(
#             self.dx * dt,
#             self.dy * dt,
#             self.dz * dt,
#             self.ddx * dt,
#             self.ddy * dt,
#             self.ddz * dt,
#         )

#     def add_acceleration(self, other: Self) -> None:
#         """Update the acceleration of the cartesian state derivative

#         Update the acceleration of this object by adding the components of the
#         acceleration vector of other cartesian state derivative. The use case is
#         to add up all the accelerations acting on a body during orbit propagation.
#         Since each acceleration is defined by a cartesian state derivative,
#         adding them up directly would also update the velocity, which is incorrect.

#         :param other: Cartesian state derivative with acceleration to add
#         """

#         # q4, q5 and q6 are the components of a general state vector that
#         # represent the acceleration in CartesianStateDerivative
#         self.q4 += other.q4
#         self.q5 += other.q5
#         self.q6 += other.q6

#     def __repr__(self) -> str:
#         return (
#             "Cartesian state derivative\n"
#             "---------------------------------\n"
#             f"dx: {self.dx}\n"
#             f"dy: {self.dy}\n"
#             f"dz: {self.dz}\n"
#             f"ddx: {self.ddx}\n"
#             f"ddy: {self.ddy}\n"
#             f"ddz: {self.ddz}"
#         )


# class _KeplerianState[U: (Double, Vector)](GenericState):
#     """Keplerian state vector

#     Single state of a time series of states of a body in a three-dimensional
#     space in terms of classical keplerian elements. Each component of the state
#     vector can be a floating point number (single state) or a numpy array of them
#     (time series of states).

#     Angles might be given in degrees or radians, but are always converted to
#     radians during initialization. For each angle, the method KeplerianState.angle
#     will return the value in radians, while KeplerianState.angle_deg will return
#     it in degrees.

#     Angles can be wrapped if specified by the wrap parameter, which is activated
#     by default. Wrapping will limit inclinations to the range [0, pi) and the
#     rest of the angles to [0, 2pi).

#     :param a: Semi-major axis [m]
#     :param e: Eccentricity
#     :param i: Inclination [rad]
#     :param raan: Right ascension of the ascending node [rad]
#     :param aop: Argument of periapsis [rad]
#     :param ta: True anomaly [rad]
#     :param deg: Angles are given in degrees (True) or radians (False)
#     :param wrap: Wrap angles (True) or not (False)
#     """

#     def __init__(
#         self,
#         a: U,
#         e: U,
#         i: U,
#         raan: U,
#         aop: U,
#         ta: U,
#         deg: bool = False,
#         wrap: bool = True,
#     ) -> None:
#         super().__init__(a, e, i, raan, aop, ta)

#         if deg:
#             self.q3 = np.deg2rad(self.q3)
#             self.q4 = np.deg2rad(self.q4)
#             self.q5 = np.deg2rad(self.q5)
#             self.q6 = np.deg2rad(self.q6)

#         if wrap:
#             self.q3 = self.q3 % np.pi
#             self.q4 = self.q4 % (2.0 * np.pi)
#             self.q5 = self.q5 % (2.0 * np.pi)
#             self.q6 = self.q6 % (2.0 * np.pi)

#         return None

#     @property
#     def a(self) -> U:
#         return self.q1[0] if self.scalar else self.q1

#     @property
#     def e(self) -> U:
#         return self.q2[0] if self.scalar else self.q2

#     @property
#     def i(self) -> U:
#         return self.q3[0] if self.scalar else self.q3

#     @property
#     def raan(self) -> U:
#         return self.q4[0] if self.scalar else self.q4

#     @property
#     def aop(self) -> U:
#         return self.q5[0] if self.scalar else self.q5

#     @property
#     def ta(self) -> U:
#         return self.q6[0] if self.scalar else self.q6

#     @property
#     def i_deg(self) -> U:
#         out: Any = np.rad2deg(self.i)
#         return out

#     @property
#     def raan_deg(self) -> U:
#         out: Any = np.rad2deg(self.raan)
#         return out

#     @property
#     def aop_deg(self) -> U:
#         out: Any = np.rad2deg(self.aop)
#         return out

#     @property
#     def ta_deg(self) -> U:
#         out: Any = np.rad2deg(self.ta)
#         return out

#     @property
#     def E(self) -> U:
#         """Eccentric anomaly"""
#         out: Any = 2.0 * np.arctan2(
#             np.sqrt(1.0 - self.e) * np.sin(0.5 * self.ta),
#             np.sqrt(1.0 + self.e) * np.cos(0.5 * self.ta),
#         )
#         return out

#     @property
#     def M(self) -> U:
#         """Mean anomaly"""
#         out: Any = self.E - self.e * np.sin(self.E)
#         return out

#     def period(self, mu: Double) -> U:
#         """Orbital period"""
#         out: Any = 2.0 * np.pi * np.sqrt(self.a * self.a * self.a / mu)
#         return out

#     def __sub__(self, other: Self | Double | int) -> Self:
#         print(
#             "WARNING: Keplerian state subtraction is not properly tested\n"
#             "It might produce unexpected results"
#         )

#         if not isinstance(other, KeplerianState):
#             return NotImplemented

#         di = np.unwrap(self.q3, period=180.0) - np.unwrap(other.q3, period=180.0)
#         draan = np.unwrap(self.q4, period=360.0) - np.unwrap(other.q4, period=360.0)
#         daop = np.unwrap(self.q5, period=360.0) - np.unwrap(other.q5, period=360.0)
#         dta = np.unwrap(self.q6, period=360.0) - np.unwrap(other.q6, period=360.0)

#         return type(self)(
#             self.q1 - other.q1, self.q2 - other.q2, di, draan, daop, dta, wrap=False
#         )

#     def __repr__(self) -> str:
#         return (
#             "Keplerian state vector\n"
#             "----------------------\n"
#             f"a: {self.a}\n"
#             f"e: {self.e}\n"
#             f"i: {self.i}\n"
#             f"RAAN: {self.raan}\n"
#             f"AoP: {self.aop}\n"
#             f"TA: {self.ta}"
#         )

#     def as_cartesian(self, mu: Double) -> "CartesianState":
#         """Conversion to cartesian state vector

#         Algorithm: Practical astrodynamics slides

#         :param mu: Gravitational parameter of central body [m^3/s^2]
#         """

#         # Auxiliary trigonometric relations
#         cos_Omega = np.cos(self.raan)
#         sin_Omega = np.sin(self.raan)
#         cos_omega = np.cos(self.aop)
#         sin_omega = np.sin(self.aop)
#         cos_i = np.cos(self.i)
#         sin_i = np.sin(self.i)
#         cos_theta = np.cos(self.ta)
#         sin_theta = np.sin(self.ta)

#         l1 = cos_Omega * cos_omega - sin_Omega * sin_omega * cos_i
#         l2 = -cos_Omega * sin_omega - sin_Omega * cos_omega * cos_i
#         m1 = sin_Omega * cos_omega + cos_Omega * sin_omega * cos_i
#         m2 = -sin_Omega * sin_omega + cos_Omega * cos_omega * cos_i
#         n1 = sin_omega * sin_i
#         n2 = cos_omega * sin_i

#         # Orbital radius
#         r = self.a * (1.0 - self.e * self.e) / (1.0 + self.e * cos_theta)

#         # Position in orbital plane
#         xi = r * cos_theta
#         eta = r * sin_theta

#         # Position in 3D space
#         x = l1 * xi + l2 * eta
#         y = m1 * xi + m2 * eta
#         z = n1 * xi + n2 * eta

#         # Angular momentum
#         H = np.sqrt(mu * self.a * (1.0 - self.e * self.e))

#         # Velocity
#         common = mu / H
#         e_cos_theta = self.e + cos_theta

#         dx = common * (l2 * e_cos_theta - l1 * sin_theta)
#         dy = common * (m2 * e_cos_theta - m1 * sin_theta)
#         dz = common * (n2 * e_cos_theta - n1 * sin_theta)

#         return CartesianState(x, y, z, dx, dy, dz)


# class _KeplerianStateDerivative[U: (Double, Vector)](GenericState):
#     """Derivative of keplerian state

#     Derivative of a single state of a time series of states of a body in a
#     three-dimensional space in terms of classical keplerian elements. Each
#     component can be a floating point number (single state) or a numpy array
#     of them (time series of states).

#     Angular velocities might be given in degrees or radians per second, but are
#     always converted to radians per second during initialization. For each angular
#     velocity, the method KeplerianStateDerivative.angvel will return the value in
#     radians per second, while KeplerianStateDerivative.angvel_deg will return it
#     in degrees per second.

#     :param da: Rate of change of semi-major axis [m/s]
#     :param de: Rate of change of eccentricity [1/s]
#     :param di: Rate of change of inclination [rad/s]
#     :param draan: Rate of change of right ascension of the ascending node [rad/s]
#     :param daop: Rate of change of argument of periapsis [rad/s]
#     :param dta: Rate of change of true anomaly [rad/s]
#     :param deg: Angular velocities are given in degrees per second (True) or
#         radians  per second (False)
#     """

#     def __init__(
#         self,
#         da: U,
#         de: U,
#         di: U,
#         draan: U,
#         daop: U,
#         dta: U,
#         deg: bool = True,
#     ) -> None:
#         super().__init__(da, de, di, draan, daop, dta)

#         if not deg:
#             self.q3 = np.rad2deg(self.q3)
#             self.q4 = np.rad2deg(self.q4)
#             self.q5 = np.rad2deg(self.q5)
#             self.q6 = np.rad2deg(self.q6)

#         return None

#     @property
#     def da(self) -> U:
#         return self.q1[0] if self.scalar else self.q1

#     @property
#     def de(self) -> U:
#         return self.q2[0] if self.scalar else self.q2

#     @property
#     def di(self) -> U:
#         return self.q3[0] if self.scalar else self.q3

#     @property
#     def draan(self) -> U:
#         return self.q4[0] if self.scalar else self.q4

#     @property
#     def daop(self) -> U:
#         return self.q5[0] if self.scalar else self.q5

#     @property
#     def dta(self) -> U:
#         return self.q6[0] if self.scalar else self.q6

#     def times_dt(self, dt: Double) -> KeplerianState:
#         """Change in keplerian state over interval dt

#         Returns the change of a keplerian state with this object as derivative
#         over a time interval dt. The main use case are numerical integrators,
#         in which the state is updated by multiplying its derivative by the
#         time step.

#         :param dt: Time step
#         :return: Change in keplerian elements over interval dt
#         """

#         return KeplerianState(
#             self.da * dt,
#             self.de * dt,
#             self.di * dt,
#             self.draan * dt,
#             self.daop * dt,
#             self.dta * dt,
#         )

#     def __repr__(self) -> str:
#         return (
#             "Keplerian state derivative\n"
#             "---------------------------------\n"
#             f"da: {self.da}\n"
#             f"de: {self.de}\n"
#             f"di: {self.di}\n"
#             f"dRAAN: {self.draan}\n"
#             f"dAoP: {self.daop}\n"
#             f"dTA: {self.dta}"
#         )


# # Convencience interfaces


# class _CartesianPosition[U: (Double, Vector)](GenericState):
#     """Three dimensional position vector in cartesian coordinates

#     Components might be floating point numbers (single vector) or numpy arrays
#     of them (time series of vectors).

#     :param x: X coordinate [m]
#     :param y: Y coordinate [m]
#     :param z: Z coordinate [m]
#     """

#     def __init__(self, x: U, y: U, z: U) -> None:

#         if isinstance(x, Double):
#             null = 0.0
#         else:
#             null = np.zeros_like(x)

#         super().__init__(x, y, z, null, null, null)
#         return None

#     @property
#     def x(self) -> U:
#         return self.q1[0] if self.scalar else self.q1

#     @property
#     def y(self) -> U:
#         return self.q2[0] if self.scalar else self.q2

#     @property
#     def z(self) -> U:
#         return self.q3[0] if self.scalar else self.q3

#     def asarray(self) -> Vector:
#         return np.array([self.x, self.y, self.z])

#     @property
#     def mag(self) -> U:
#         return np.linalg.norm(self.asarray(), axis=0)

#     def __repr__(self) -> str:
#         return (
#             "Position vector: \n"
#             "-----------------\n"
#             f"x: {self.x}\n"
#             f"y: {self.y}\n"
#             f"z: {self.z}\n"
#         )


# class _CartesianVelocity[U: (Double, Vector)](GenericState):
#     """Three dimensional velocity vector in cartesian coordinates

#     Components might be floating point numbers (single vector) or numpy arrays
#     of them (time series of vectors).

#     :param dx: X component of the velocity vector [m/s]
#     :param dy: Y component of the velocity vector [m/s]
#     :param dz: Z component of the velocity vector [m/s]
#     """

#     def __init__(self, dx: U, dy: U, dz: U) -> None:

#         if isinstance(dx, Double):
#             null = 0.0
#         else:
#             null = np.zeros_like(dx)

#         super().__init__(null, null, null, dx, dy, dz)
#         return None

#     @property
#     def dx(self) -> U:
#         return self.q4[0] if self.scalar else self.q4

#     @property
#     def dy(self) -> U:
#         return self.q5[0] if self.scalar else self.q5

#     @property
#     def dz(self) -> U:
#         return self.q6[0] if self.scalar else self.q6

#     def asarray(self) -> Vector:
#         return np.array([self.dx, self.dy, self.dz])

#     @property
#     def mag(self) -> U:
#         return np.linalg.norm(self.asarray(), axis=0)

#     def __repr__(self) -> str:
#         return (
#             "Velocity vector: \n"
#             "-----------------\n"
#             f"dx: {self.dx}\n"
#             f"dy: {self.dy}\n"
#             f"dz: {self.dz}\n"
#         )


# class _SphericalPosition[U: (Double, Vector)](GenericState):
#     """Three dimensional position vector in spherical coordinates

#     Components might be floating point numbers (single vector) or numpy arrays
#     of them (time series of vectors).

#     Angles can be passed in radians (default) or degrees (deg=True), but they
#     are always converted to radians during initialization. For each angle, the
#     method SphericalPosition.angle will return the value in radians, while
#     SphericalPosition.angle_deg will return it in degrees.

#     :param r: Radial distance [m]
#     :param theta: Horizontal angular distance [rad / deg]
#     :param phi: Vertical angular distance [rad / deg]
#     :param deg: Angles are given in degrees (True) or radians (False)
#     :param wrap: Wrap angles (True) or not (False)
#     """

#     def __init__(
#         self, r: U, theta: U, phi: U, deg: bool = False, wrap: bool = False
#     ) -> None:

#         if isinstance(r, Double):
#             null = 0.0
#         else:
#             null = np.zeros_like(r)

#         super().__init__(r, theta, phi, null, null, null)

#         if deg:
#             self.q2 = np.deg2rad(self.q2)
#             self.q3 = np.deg2rad(self.q3)
#         if wrap:
#             raise NotImplementedError(
#                 "Angle wrapping is not implemented for spherical coordinates"
#             )

#         return None

#     @property
#     def r(self) -> U:
#         return self.q1[0] if self.scalar else self.q1

#     @property
#     def theta(self) -> U:
#         return self.q2[0] if self.scalar else self.q2

#     @property
#     def phi(self) -> U:
#         return self.q3[0] if self.scalar else self.q3

#     @property
#     def theta_deg(self) -> U:
#         out: Any = np.rad2deg(self.theta)
#         return out

#     @property
#     def phi_deg(self) -> U:
#         out: Any = np.rad2deg(self.phi)
#         return out

#     def height(self, reference: Double) -> U:
#         return self.r - reference

#     def asarray(self) -> Vector:
#         return np.array([self.r, self.theta, self.phi])

#     def __repr__(self) -> str:
#         return (
#             "Spherical position vector\n"
#             "------------------------\n"
#             f"r: {self.r}\n"
#             f"theta: {self.theta}\n"
#             f"phi: {self.phi}\n"
#         )
