import numpy as np
from typing import Self, Any, Iterator
from numpy.typing import NDArray
from datetime import datetime

Double = np.floating | float
Vector = NDArray[np.floating]
Date = datetime

"""
NOTE: Updated implementation of cartesian and keplerian states and their
derivatives. Custom date type is replaced by datetime.datetime and the new
implementations for the types break backwards compatibility, so the rest of
the library must be updated before these can be used.
"""


class GenericState[T: (Double, Vector)]:
    """Base class for state vectors

    Implements operations and basic functionality that is common to all types
    of state vectors and implements the containers that are used internally
    to manipulate their components. The elements q1 to q6 are representations
    of the six components of a state vector as numpy arrays.

    This class is not meant to be instantiated directly, but to be subclassed
    to create specific types of state vectors: cartesian, keplerian, etc.
    """

    def __init__(self, q1: T, q2: T, q3: T, q4: T, q5: T, q6: T) -> None:

        if isinstance(q1, Double):
            self.scalar = True
            input = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float64)
            assert len(input.shape) == 1
            assert input.size == 6
        elif isinstance(q1, np.ndarray) and isinstance(q1[0], Double):
            self.scalar = False
            try:
                input = np.array([q1, q2, q3, q4, q5, q6], dtype=np.float64)
                assert len(input.shape) == 2
                assert input.size / 6 == input.shape[1]
            except ValueError or AssertionError:
                raise ValueError(
                    "Components of state vector cannot have different sizes"
                )
        else:
            raise TypeError(
                "Components of state vector must be floating point numbers"
                " or numpy arrays"
            )

        # Initialize state components
        self.q1: Any = np.array([q1], dtype=np.float64).ravel()
        self.q2: Any = np.array([q2], dtype=np.float64).ravel()
        self.q3: Any = np.array([q3], dtype=np.float64).ravel()
        self.q4: Any = np.array([q4], dtype=np.float64).ravel()
        self.q5: Any = np.array([q5], dtype=np.float64).ravel()
        self.q6: Any = np.array([q6], dtype=np.float64).ravel()

        return None

    @property
    def size(self) -> int:
        return self.q1.size

    def __getitem__(self, index: int | slice) -> Self:

        return type(self)(
            self.q1[index],
            self.q2[index],
            self.q3[index],
            self.q4[index],
            self.q5[index],
            self.q6[index],
        )

    def __iter__(self) -> Iterator[Self]:

        return iter([self.__getitem__(idx) for idx in range(self.size)])

    def __add__(self, other: Self | Double | int) -> Self:

        if isinstance(other, Double) or isinstance(other, int):
            other = float(other)
            return type(self)(
                self.q1 + other,
                self.q2 + other,
                self.q3 + other,
                self.q4 + other,
                self.q5 + other,
                self.q6 + other,
            )
        elif isinstance(other, Self):
            return type(self)(
                self.q1 + other.q1,
                self.q2 + other.q2,
                self.q3 + other.q3,
                self.q4 + other.q4,
                self.q5 + other.q5,
                self.q6 + other.q6,
            )
        else:
            raise TypeError(
                "Addition is only supported for state vectors and scalars",
            )

    def __sub__(self, other: Self | Double | int) -> Self:

        if isinstance(other, Double) or isinstance(other, int):
            other = float(other)
            return type(self)(
                self.q1 - other,
                self.q2 - other,
                self.q3 - other,
                self.q4 - other,
                self.q5 - other,
                self.q6 - other,
            )
        elif isinstance(other, Self):
            return type(self)(
                self.q1 - other.q1,
                self.q2 - other.q2,
                self.q3 - other.q3,
                self.q4 - other.q4,
                self.q5 - other.q5,
                self.q6 - other.q6,
            )
        else:
            raise TypeError(
                "Subtraction is only supported for state vectors and scalars"
            )

    def __mul__(self, other: Self | Double | int) -> Self:

        if isinstance(other, Double) or isinstance(other, int):
            other = float(other)
            return type(self)(
                self.q1 * other,
                self.q2 * other,
                self.q3 * other,
                self.q4 * other,
                self.q5 * other,
                self.q6 * other,
            )
        elif isinstance(other, Self):
            return type(self)(
                self.q1 * other.q1,
                self.q2 * other.q2,
                self.q3 * other.q3,
                self.q4 * other.q4,
                self.q5 * other.q5,
                self.q6 * other.q6,
            )
        else:
            raise TypeError(
                "Multiplication is only supported for state vectors and scalars"
            )

    def __eq__(self, other) -> bool:

        if isinstance(other, Double) or isinstance(other, int):
            other = float(other)
            return (
                self.q1 == other
                and self.q2 == other
                and self.q3 == other
                and self.q4 == other
                and self.q5 == other
                and self.q6 == other
            )
        elif isinstance(other, Self):
            return (
                self.q1 == other.q1
                and self.q2 == other.q2
                and self.q3 == other.q3
                and self.q4 == other.q4
                and self.q5 == other.q5
                and self.q6 == other.q6
            )
        else:
            raise TypeError(
                "Can only check for equality against state vectors and scalars"
            )

    def append(self, other: Self | Double | int) -> Self:

        if isinstance(other, Double) or isinstance(other, int):
            other = float(other)
            return type(self)(
                np.append(self.q1, other),
                np.append(self.q2, other),
                np.append(self.q3, other),
                np.append(self.q4, other),
                np.append(self.q5, other),
                np.append(self.q6, other),
            )
        elif isinstance(other, Self):
            return type(self)(
                np.append(self.q1, other.q1),
                np.append(self.q2, other.q2),
                np.append(self.q3, other.q3),
                np.append(self.q4, other.q4),
                np.append(self.q5, other.q5),
                np.append(self.q6, other.q6),
            )
        else:
            raise TypeError("Append is only supported for state vectors and scalars")

    def asarray(self) -> Vector:

        return np.array(
            [self.q1, self.q2, self.q3, self.q4, self.q5, self.q6], dtype=np.float64
        )


class CartesianState[T: (Double, Vector)](GenericState):
    """Cartesian state vector

    Single state of a time series of states of a body in a three-dimensional
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

    def __init__(self, x: T, y: T, z: T, dx: T, dy: T, dz: T) -> None:
        super().__init__(x, y, z, dx, dy, dz)
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

    @property
    def dx(self) -> T:
        return self.q4[0] if self.scalar else self.q4

    @property
    def dy(self) -> T:
        return self.q5[0] if self.scalar else self.q5

    @property
    def dz(self) -> T:
        return self.q6[0] if self.scalar else self.q6

    @property
    def r_vec(self) -> Vector:
        return np.array([self.x, self.y, self.z])

    @property
    def v_vec(self) -> Vector:
        return np.array([self.dx, self.dy, self.dz])

    @property
    def r_mag(self) -> T:
        return np.linalg.norm(self.r_vec, axis=0)

    @property
    def v_mag(self) -> T:
        return np.linalg.norm(self.v_vec, axis=0)

    def __repr__(self) -> str:

        return (
            "Cartesian state vector\n"
            "----------------------\n"
            f"x: {self.x}\n"
            f"y: {self.y}\n"
            f"z: {self.z}\n"
            f"dx: {self.dx}\n"
            f"dy: {self.dy}\n"
            f"dz: {self.dz}"
        )


class CartesianStateDerivative[T: (Double, Vector)](GenericState):
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

    def __init__(self, dx: T, dy: T, dz: T, ddx: T, ddy: T, ddz: T) -> None:
        super().__init__(dx, dy, dz, ddx, ddy, ddz)
        return None

    @property
    def dx(self) -> T:
        return self.q1[0] if self.scalar else self.q1

    @property
    def dy(self) -> T:
        return self.q2[0] if self.scalar else self.q2

    @property
    def dz(self) -> T:
        return self.q3[0] if self.scalar else self.q3

    @property
    def ddx(self) -> T:
        return self.q4[0] if self.scalar else self.q4

    @property
    def ddy(self) -> T:
        return self.q5[0] if self.scalar else self.q5

    @property
    def ddz(self) -> T:
        return self.q6[0] if self.scalar else self.q6

    @property
    def v_vec(self) -> Vector:
        return np.array([self.dx, self.dy, self.dz])

    @property
    def a_vec(self) -> Vector:
        return np.array([self.ddx, self.ddy, self.ddz])

    @property
    def v_mag(self) -> T:
        return np.linalg.norm(self.v_vec, axis=0)

    @property
    def dv_mag(self) -> T:
        return np.linalg.norm(self.v_vec, axis=0)

    def times_dt(self, dt: Double) -> CartesianState:
        """Change in cartesian state over interval dt

        Returns the change of a cartesian state with this object as derivative
        over a time interval dt. The main use case are numerical integrators,
        in which the state is updated by multiplying its derivative by the
        time step.

        :param dt: Time step
        :return: Change in cartesian elements over interval dt
        """

        return CartesianState(
            self.dx * dt,
            self.dy * dt,
            self.dz * dt,
            self.ddx * dt,
            self.ddy * dt,
            self.ddz * dt,
        )

    def add_acceleration(self, other: Self) -> None:
        """Update the acceleration of the cartesian state derivative

        Update the acceleration of this object by adding the components of the
        acceleration vector of other cartesian state derivative. The use case is
        to add up all the accelerations acting on a body during orbit propagation.
        Since each acceleration is defined by a cartesian state derivative,
        adding them up directly would also update the velocity, which is incorrect.

        :param other: Cartesian state derivative with acceleration to add
        """

        # q4, q5 and q6 are the components of a general state vector that
        # represent the acceleration in CartesianStateDerivative
        self.q4 += other.q4
        self.q5 += other.q5
        self.q6 += other.q6

    def __repr__(self) -> str:

        return (
            "Cartesian state derivative\n"
            "---------------------------------\n"
            f"dx: {self.dx}\n"
            f"dy: {self.dy}\n"
            f"dz: {self.dz}\n"
            f"ddx: {self.ddx}\n"
            f"ddy: {self.ddy}\n"
            f"ddz: {self.ddz}"
        )


class KeplerianState[T: (Double, Vector)](GenericState):
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

    def __init__(
        self,
        a: T,
        e: T,
        i: T,
        raan: T,
        aop: T,
        ta: T,
        deg: bool = True,
        wrap: bool = True,
    ) -> None:

        super().__init__(a, e, i, raan, aop, ta)

        if deg:
            self.q3 = np.deg2rad(self.q3)
            self.q4 = np.deg2rad(self.q4)
            self.q5 = np.deg2rad(self.q5)
            self.q6 = np.deg2rad(self.q6)

        if wrap:
            self.q3 = self.q3 % np.pi
            self.q4 = self.q4 % (2.0 * np.pi)
            self.q5 = self.q5 % (2.0 * np.pi)
            self.q6 = self.q6 % (2.0 * np.pi)

        return None

    @property
    def a(self) -> T:
        return self.q1[0] if self.scalar else self.q1

    @property
    def e(self) -> T:
        return self.q2[0] if self.scalar else self.q2

    @property
    def i(self) -> T:
        return self.q3[0] if self.scalar else self.q3

    @property
    def raan(self) -> T:
        return self.q4[0] if self.scalar else self.q4

    @property
    def aop(self) -> T:
        return self.q5[0] if self.scalar else self.q5

    @property
    def ta(self) -> T:
        return self.q6[0] if self.scalar else self.q6

    @property
    def i_deg(self) -> T:
        out: Any = np.rad2deg(self.i)
        return out

    @property
    def raan_deg(self) -> T:
        out: Any = np.rad2deg(self.raan)
        return out

    @property
    def aop_deg(self) -> T:
        out: Any = np.rad2deg(self.aop)
        return out

    @property
    def ta_deg(self) -> T:
        out: Any = np.rad2deg(self.ta)
        return out

    def __sub__(self, other: Self | Double | int) -> Self:

        print(
            "WARNING: Keplerian state subtraction is not properly tested\n"
            "It might produce unexpected results"
        )

        if not isinstance(other, KeplerianState):
            return NotImplemented

        di = np.unwrap(self.q3, period=180.0) - np.unwrap(other.q3, period=180.0)
        draan = np.unwrap(self.q4, period=360.0) - np.unwrap(other.q4, period=360.0)
        daop = np.unwrap(self.q5, period=360.0) - np.unwrap(other.q5, period=360.0)
        dta = np.unwrap(self.q6, period=360.0) - np.unwrap(other.q6, period=360.0)

        return type(self)(
            self.q1 - other.q1, self.q2 - other.q2, di, draan, daop, dta, wrap=False
        )

    def __repr__(self) -> str:

        return (
            "Keplerian state vector\n"
            "----------------------\n"
            f"a: {self.a}\n"
            f"e: {self.e}\n"
            f"i: {self.i}\n"
            f"RAAN: {self.raan}\n"
            f"AoP: {self.aop}\n"
            f"TA: {self.ta}"
        )


class KeplerianStateDerivative[T: (Double, Vector)](GenericState):
    """Derivative of keplerian state

    Derivative of a single state of a time series of states of a body in a
    three-dimensional space in terms of classical keplerian elements. Each
    component can be a floating point number (single state) or a numpy array
    of them (time series of states).

    Angular velocities might be given in degrees or radians per second, but are
    always converted to radians per second during initialization. For each angular
    velocity, the method KeplerianStateDerivative.angvel will return the value in
    radians per second, while KeplerianStateDerivative.angvel_deg will return it
    in degrees per second.

    :param da: Rate of change of semi-major axis [m/s]
    :param de: Rate of change of eccentricity [1/s]
    :param di: Rate of change of inclination [rad/s]
    :param draan: Rate of change of right ascension of the ascending node [rad/s]
    :param daop: Rate of change of argument of periapsis [rad/s]
    :param dta: Rate of change of true anomaly [rad/s]
    :param deg: Angular velocities are given in degrees per second (True) or
        radians  per second (False)
    """

    def __init__(
        self,
        da: T,
        de: T,
        di: T,
        draan: T,
        daop: T,
        dta: T,
        deg: bool = True,
    ) -> None:

        super().__init__(da, de, di, draan, daop, dta)

        if not deg:
            self.q3 = np.rad2deg(self.q3)
            self.q4 = np.rad2deg(self.q4)
            self.q5 = np.rad2deg(self.q5)
            self.q6 = np.rad2deg(self.q6)

        return None

    @property
    def da(self) -> T:
        return self.q1[0] if self.scalar else self.q1

    @property
    def de(self) -> T:
        return self.q2[0] if self.scalar else self.q2

    @property
    def di(self) -> T:
        return self.q3[0] if self.scalar else self.q3

    @property
    def draan(self) -> T:
        return self.q4[0] if self.scalar else self.q4

    @property
    def daop(self) -> T:
        return self.q5[0] if self.scalar else self.q5

    @property
    def dta(self) -> T:
        return self.q6[0] if self.scalar else self.q6

    def times_dt(self, dt: Double) -> KeplerianState:
        """Change in keplerian state over interval dt

        Returns the change of a keplerian state with this object as derivative
        over a time interval dt. The main use case are numerical integrators,
        in which the state is updated by multiplying its derivative by the
        time step.

        :param dt: Time step
        :return: Change in keplerian elements over interval dt
        """

        return KeplerianState(
            self.da * dt,
            self.de * dt,
            self.di * dt,
            self.draan * dt,
            self.daop * dt,
            self.dta * dt,
        )

    def __repr__(self) -> str:

        return (
            "Keplerian state derivative\n"
            "---------------------------------\n"
            f"da: {self.da}\n"
            f"de: {self.de}\n"
            f"di: {self.di}\n"
            f"dRAAN: {self.draan}\n"
            f"dAoP: {self.daop}\n"
            f"dTA: {self.dta}"
        )
