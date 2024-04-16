from .core import Double, Vector, T
from ..frames import ICRF, Frame
import numpy as np
from typing import Any, Generic
from ..constants import pi, twopi, halfpi

# THIS IS THE WIP VERSION OF THE MODULE (TO BE BUILT FROM STATE_WRONG AND STATE)


class GenericState(Generic[T]):
    """Base class for state vectors"""

    def __init__(
        self,
        q1: T,
        q2: T,
        q3: T,
        q4: T,
        q5: T,
        q6: T,
        frame: type[Frame] = ICRF,
        deg: bool = False,
        wrap: bool = True,
        _components: dict[str, str] | None = None,
        _angles: dict[str, tuple[Double, Double]] | None = None,
    ) -> None:

        # Get properties
        if _components:
            self.components = _components
        else:
            self.components = {
                "q1": "q1",
                "q2": "q2",
                "q3": "q3",
                "q4": "q4",
                "q5": "q5",
                "q6": "q6",
            }
        # Get angles and boundaries
        if _angles:
            self.angles = _angles
        else:
            self.angles: dict[str, tuple[Double, Double]] = {}
        self.has_angles = len(self.angles) > 0

        # Check input
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

        # Initialize frame and state components
        self.frame = frame
        self.q1: Any = np.array([q1], dtype=np.float64).ravel()
        self.q2: Any = np.array([q2], dtype=np.float64).ravel()
        self.q3: Any = np.array([q3], dtype=np.float64).ravel()
        self.q4: Any = np.array([q4], dtype=np.float64).ravel()
        self.q5: Any = np.array([q5], dtype=np.float64).ravel()
        self.q6: Any = np.array([q6], dtype=np.float64).ravel()

        if not self.has_angles:
            return None

        # Convert angles to radians if necessary
        if deg:
            for angle in self.angles:
                self.__dict__[angle] = np.deg2rad(self.__dict__[angle])

        # Wrap angles if necessary
        if wrap:
            for angle, (low, high) in self.angles.items():
                self.__dict__[angle] = self.wrap_angle(self.__dict__[angle], low, high)

        return None

    @staticmethod
    def wrap_angle(angle: T, low: Double = 0.0, high: Double = twopi) -> T:
        """Wrap angle to [low, high] interval"""

        # TERRIBLY INEFFICIENT, BUT IT WORKS

        angle %= twopi
        if low == 0.0 and high == twopi:
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

        raise ValueError("Invalid angle boundaries")
