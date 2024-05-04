from .base import Force
from .. import types as nt


class PointMass(Force):

    def __init__(self, mu: nt.Double) -> None:

        self.mu = mu

        return None

    def __call__(
        self, t: nt.Double, s: nt.CartesianState, fr: nt.Double = 0.0
    ) -> nt.CartesianStateDerivative:

        f = -self.mu / (s.r_mag * s.r_mag * s.r_mag)

        return nt.CartesianStateDerivative(s.dx, s.dy, s.dz, f * s.x, f * s.y, f * s.z)
