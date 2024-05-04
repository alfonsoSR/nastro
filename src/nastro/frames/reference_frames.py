from typing import TypeVar, Generic
import numpy as np
from ..types.core import Vector, Double
from ..constants import miliarcsec

Frame = TypeVar("Frame", bound="ReferenceFrame")


class ReferenceFrame:
    """Base class for reference frames"""

    variables = {"r": "q1", "theta": "q2", "phi": "q3"}

    @classmethod
    def convert(
        cls, q1: Vector, q2: Vector, q3: Vector, frame: type[Frame]
    ) -> tuple[Vector, Vector, Vector]:
        raise NotImplementedError


class ICRF(ReferenceFrame):

    variables = {"r": "q1", "alpha": "q2", "delta": "q3"}

    dalpha0 = -14.6 * miliarcsec
    xi0 = -16.6170 * miliarcsec
    eta0 = -6.8192 * miliarcsec

    B = np.array(
        [
            [
                1.0 - 0.5 * (dalpha0 * dalpha0 + xi0 * xi0),
                dalpha0,
                -xi0,
            ],
            [
                -dalpha0 - eta0 * xi0,
                1.0 - 0.5 * (dalpha0 * dalpha0 + eta0 * eta0),
                -eta0,
            ],
            [
                xi0 - eta0 * dalpha0,
                eta0 + xi0 * dalpha0,
                1.0 - 0.5 * (xi0 * xi0 + eta0 * eta0),
            ],
        ]
    )

    @classmethod
    def convert(
        cls, q1: Vector, q2: Vector, q3: Vector, frame: type[Frame]
    ) -> tuple[Vector, Vector, Vector]:
        if frame == J2000:
            return np.dot(cls.B, np.array([q1, q2, q3]))
        else:
            raise NotImplementedError


class J2000(ReferenceFrame):

    variables = {"r": "q1", "theta": "q2", "phi": "q3"}

    @classmethod
    def convert(
        cls, q1: Vector, q2: Vector, q3: Vector, frame: type[Frame]
    ) -> tuple[Vector, Vector, Vector]:

        if frame == ICRF:
            return np.dot(ICRF.B.T, np.array([q1, q2, q3]))
        else:
            raise NotImplementedError
