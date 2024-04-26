from .base import Force
from ..types import CartesianState, CartesianStateDerivative, JulianDay, Double


class PointMass(Force):

    def __init__(self, mu: Double) -> None:

        self.mu = mu

        return None

    def __call__(
        self, t: Double, s: CartesianState, fr: Double = 0.0
    ) -> CartesianStateDerivative:

        f = -self.mu / (s.r_mag * s.r_mag * s.r_mag)

        return CartesianStateDerivative(s.dx, s.dy, s.dz, f * s.x, f * s.y, f * s.z)
