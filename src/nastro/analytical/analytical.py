from ..types import KeplerianState, Vector, time_to_mean_anomaly, Double
from ..types.conversion import eccentric_to_true_anomaly
import numpy as np
from .enrke import enrke
from ..constants import twopi


def keplerian_orbit(s0: KeplerianState, epochs: Vector, mu: Double) -> KeplerianState:

    if not s0.scalar:
        raise ValueError("Initial state must be scalar")

    # Initial value of mean anomaly
    M = time_to_mean_anomaly(epochs - epochs[0], s0.a, mu, s0.M)
    E = enrke(M, s0.e)
    nu_wrapped = eccentric_to_true_anomaly(E, s0.e)
    nu = np.unwrap(nu_wrapped, period=twopi)
    if not np.allclose(nu[0], s0.ta, atol=1e-15, rtol=0.0):
        raise ValueError(f"True anomaly unwrapping failed: {nu[0]}, {s0.ta}")

    # Generate keplerian state
    base = np.ones_like(epochs, dtype=np.float64)
    return KeplerianState(
        s0.a * base,
        s0.e * base,
        s0.i * base,
        s0.raan * base,
        s0.aop * base,
        nu,
    )


# def keplerian_orbit(s0: KeplerianState, epochs: Vector, mu: float) -> KeplerianState:
#     """Generate an ideal keplerian trajectory

#     :param s0: Keplerian elements at initial epoch
#     :param epochs: Epoch at which to calculate the state [s]
#     :param mu: Standard gravitational parameter of central body [m^3/s^2]
#     """

#     a = np.ones_like(epochs) * s0.a
#     e = np.ones_like(epochs) * s0.e
#     i = np.ones_like(epochs) * s0.i
#     Omega = np.ones_like(epochs) * s0.Omega
#     omega = np.ones_like(epochs) * s0.omega

#     # Calculate mean anomaly as function of time
#     # M = (t - t0) * sqrt(mu / a^3)
#     M = (epochs - epochs[0]) * np.sqrt(mu / (s0.a * s0.a * s0.a))

#     # Calculate eccentric anomaly from mean anomaly
#     # M = E - e * sin(E)
#     # Newton-Raphson using M as initial guess
#     def f(E, e, M):
#         return E - e * np.sin(E) - M

#     def fprime(E, e, M):
#         return 1 - e * np.cos(E)

#     def fprime2(E, e, M):
#         return e * np.sin(E)

#     guess_E = M

#     E = newton(f, guess_E, fprime, fprime2=fprime2, args=(e, M))

#     # Calculate true anomaly from eccentric anomaly
#     nu = np.rad2deg(2.0 * np.arctan(np.sqrt((1 + e) / (1 - e)) * np.tan(E / 2)))

#     return KeplerianState(a, e, i, Omega, omega, nu)
