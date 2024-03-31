from .core import Vector, Double
from typing import Any
import numpy as np


def time_to_mean_anomaly[T](time: T, a: Double, mu: Double, M0: Double = 0.0) -> T:
    """Mean anomaly from time past periapsis.

    :param time: Time since periapsis [s]
    :param a: Semi-major axis of the keplerian orbit [m]
    :param mu: Standard gravitational parameter of central body [m^3/s^2]
    :param M0: Initial mean anomaly [rad]
    """
    return M0 + time * np.sqrt(mu / (a * a * a))


def true_to_eccentric_anomaly[T: (Double, Vector)](ta: T, e: T) -> T:
    """Eccentric anomaly from true anomaly and eccentricity

    :param theta: True anomaly [rad]
    :param e: Eccentricity of the keplerian orbit
    """
    out: Any = 2.0 * np.arctan2(
        np.sqrt(1.0 - e) * np.sin(0.5 * ta), np.sqrt(1.0 + e) * np.cos(0.5 * ta)
    )
    return out


def eccentric_to_true_anomaly[T: (Double, Vector)](E: T, e: T) -> T:
    """True anomaly from eccentric anomaly and eccentricity

    :param E: Eccentric anomaly [rad]
    :param e: Eccentricity of the keplerian orbit
    """
    out: Any = 2.0 * np.arctan2(
        np.sqrt(1.0 + e) * np.sin(0.5 * E), np.sqrt(1.0 - e) * np.cos(0.5 * E)
    )
    return out


def eccentric_to_mean_anomaly[T: (Double, Vector)](E: T, e: T) -> T:
    """Mean anomaly from eccentric anomaly and eccentricity

    :param E: Eccentric anomaly [rad]
    :param e: Eccentricity of the keplerian orbit
    """
    out: Any = E - e * np.sin(E)
    return out


# def cartesian2spherical(t: Vector, s: CartesianState) -> SphericalGeocentric:
#     """Convert cartesian state vectors to spherical coordinates.

#     :param t: Epoch in which the cartesian state is knonw or time series [JD].
#     :param s: Cartesian state [m, m/s].
#     :return: Spherical coordinates [m, deg].
#     """

#     def _constrain_theta(theta: float) -> float:
#         """Constain angle to [0, 360) deg.

#         :param theta: Angle in degrees.
#         """
#         while (theta > 360.0) or (theta < -360.0):
#             if theta > 360.0:
#                 theta -= 360.0
#             elif theta < -360.0:
#                 theta += 360.0

#         return theta

#     # Calculate Julian century wrt to J2000
#     T = (t - 2451545.0) / 36525.0

#     # Calculate GMST in radians
#     # Algorithm from Curtis
#     a0 = 100.4606184
#     a1 = 36000.77004
#     a2 = 0.000387933
#     a3 = -2.583e-8

#     theta_GMST = a0 + a1 * T + a2 * T * T + a3 * T * T * T

#     for idx, theta in enumerate(theta_GMST):
#         theta_GMST[idx] = _constrain_theta(theta)
#     theta_GMST = np.deg2rad(theta_GMST)

#     # Calculate dates from JD

#     ut_list = np.zeros_like(t)
#     for idx, jd in enumerate(t):
#         date, err = jd2date(jd, None)
#         assert isinstance(date, Date)
#         ut_list[idx] = date.hour * 3600.0 + date.minute * 60.0 + date.second + err

#     # Calculate Greenwhich sidereal time at current epoch
#     # Earth's rotation speed taken from Horizons
#     theta_G = theta_GMST + 0.00007292115 * ut_list

#     # Rotate position vector to ECI
#     x_ECI = s.x * np.cos(theta_G) + s.y * np.sin(theta_G)
#     y_ECI = -s.x * np.sin(theta_G) + s.y * np.cos(theta_G)
#     z_ECI = s.z

#     # Calculate spherical coordinates
#     r = np.sqrt(x_ECI * x_ECI + y_ECI * y_ECI + z_ECI * z_ECI)
#     lat = np.rad2deg(np.arcsin(z_ECI / r))
#     long = np.rad2deg(np.arctan2(y_ECI, x_ECI))

#     return SphericalGeocentric(r, lat, long)

#     # return SphericalCoordinates(r, lat, long)


# def cartesian2spherical_inertial(s: CartesianState) -> SphericalGeocentric:
#     """Convert cartesian state vector to spherical, inertial coordinates

#     :param s: Cartesian state [m, m/s]
#     :return: Spherical inertial coordinates [m, deg]
#     """
#     lat = np.rad2deg(np.arcsin(s.z / s.r_mag))
#     r_xy = np.sqrt(s.x * s.x + s.y * s.y)
#     lon = np.rad2deg(np.arctan2(s.y / r_xy, s.x / r_xy))

#     return SphericalGeocentric(s.r_mag, lat, lon)
