import numpy as np
from typing import Any, Iterator, Self
from ..types import Double, Vector, Scalar, is_double, is_vector, is_scalar
import traceback
from astropy import coordinates
from ..frames import Frame


class State[U: (Double, Vector)]:
    """Base class for state vectors

    Implements operations and basic functionality that is common to all types
    of state vectors and implements the containers that are used internally
    to manipulate their components. The elements q1 to q6 are representations
    of the six components of a state vector as numpy arrays.

    This class is not meant to be instantiated directly, but to be subclassed
    to create specific types of state vectors: cartesian, keplerian, etc.
    """

    __slots__ = ("q1", "q2", "q3", "q4", "q5", "q6", "q7", "frame")
    properties = {
        "q1": "q1",
        "q2": "q2",
        "q3": "q3",
        "q4": "q4",
        "q5": "q5",
        "q6": "q6",
        "q7": "q7",
        "frame": "frame",
    }
    angles: dict[str, tuple[Double, Double]] = {}

    def __init__(
        self, q1: U, q2: U, q3: U, q4: U, q5: U, q6: U, q7: U, frame: Frame
    ) -> None:

        if is_double(q1):
            _input = np.array([q1, q2, q3, q4, q5, q6, q7], dtype=np.float64)[:, None]
            assert len(_input.shape) == 2
            assert _input.shape == (7, 1)
        elif is_vector(q1):
            try:
                _input = np.array([q1, q2, q3, q4, q5, q6, q7], dtype=np.float64)
                assert len(_input.shape) == 2
                assert _input.shape[0] == 7
            except ValueError or AssertionError:
                raise ValueError(
                    f"Failed to create {self.__class__.__name__}: "
                    "Components have different sizes"
                )
        else:
            raise TypeError(
                f"Failed to create {self.__class__.__name__}: "
                "Components must be vectors or floating point numbers"
            )

        # Initialize frame and generic state components
        self.frame = frame
        self.q1 = np.array(q1, dtype=np.float64, ndmin=1)
        self.q2 = np.array(q2, dtype=np.float64, ndmin=1)
        self.q3 = np.array(q3, dtype=np.float64, ndmin=1)
        self.q4 = np.array(q4, dtype=np.float64, ndmin=1)
        self.q5 = np.array(q5, dtype=np.float64, ndmin=1)
        self.q6 = np.array(q6, dtype=np.float64, ndmin=1)
        self.q7 = np.array(q7, dtype=np.float64, ndmin=1)

        if not self.has_angles:
            return None

    @property
    def scalar(self) -> bool:
        return len(self.q1) == 1

    @property
    def has_angles(self) -> bool:
        return len(self.angles) > 0
