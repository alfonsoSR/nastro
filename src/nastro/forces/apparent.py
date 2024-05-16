from .base import Force
from .. import types as nt
import numpy as np


class Coriolis(Force):

    def __init__(self, omega: nt.Double) -> None:
        self.omega = np.array([0.0, 0.0, omega])
        return None

    def __call__(
        self, t: nt.Double, s: nt.CartesianState, fr: nt.Double = 0.0
    ) -> nt.CartesianStateDerivative:
        return nt.CartesianStateDerivative(
            s.dx, s.dy, s.dz, *(-2.0 * np.cross(self.omega, s.v_vec, axis=0))
        )


class Centrifugal(Force):

    def __init__(self, omega: nt.Double) -> None:
        self.omega = np.array([0.0, 0.0, omega])
        return None

    def __call__(
        self, t: nt.Double, s: nt.CartesianState, fr: nt.Double = 0.0
    ) -> nt.CartesianStateDerivative:
        return nt.CartesianStateDerivative(
            s.dx,
            s.dy,
            s.dz,
            *(-np.cross(self.omega, np.cross(self.omega, s.r_vec, axis=0), axis=0)),
        )
