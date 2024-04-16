from ..types import Double, Vector
from typing import Any, TypeVar
from ..catalog import Earth
import numpy as np

# from .state import CartesianPosition, SphericalPosition
from ..constants import miliarcsec

# from typing import Generic

Frame = TypeVar("Frame", bound="ReferenceFrame")


class ReferenceFrame[T: (Double, Vector)]:
    """Base class for reference frames"""

    variables = {
        "rho": "q1",
        "alpha": "q2",
        "delta": "q3",
        "drho": "q4",
        "dalpha": "q5",
        "ddelta": "q6",
    }
    angles: dict[str, tuple[float, float]] = {
        "q2": (0.0, 2 * np.pi),
        "q3": (-0.5 * np.pi, 0.5 * np.pi),
    }

    @classmethod
    def to_cartesian(cls, q1: T, q2: T, q3: T) -> tuple[T, T, T]:
        _q1: Any = q1 * np.cos(q2) * np.cos(q3)
        _q2: Any = q1 * np.sin(q2) * np.cos(q3)
        _q3: Any = q1 * np.sin(q3)
        return _q1, _q2, _q3

    @classmethod
    def to_spherical(cls, q1: T, q2: T, q3: T) -> tuple[T, T, T]:
        _q1: Any = np.sqrt(q1 * q1 + q2 * q2 + q3 * q3)
        _q2: Any = np.arctan2(q2, q1)
        _q3: Any = np.arcsin(q3 / _q1)
        return _q1, _q2, _q3

    @classmethod
    def transform(cls, q1: T, q2: T, q3: T, frame: type[Frame]) -> tuple[T, T, T]:
        """Transform cartesian vector to a different reference frame"""

        try:
            return getattr(cls, f"to_{frame.__name__.lower()}")(q1, q2, q3)
        except AttributeError:
            raise NotImplementedError(
                f"Conversion from {cls.__name__} to {frame.__name__} not implemented"
            )

    @classmethod
    def to_icrf(cls, q1: T, q2: T, q3: T) -> tuple[T, T, T]:
        raise NotImplementedError(
            "Attempted conversion to ICRF from generic reference frame"
        )

    @classmethod
    def to_j2000(cls, q1: T, q2: T, q3: T) -> tuple[T, T, T]:
        raise NotImplementedError(
            "Attempted conversion to J2000 from generic reference frame"
        )


class ICRF[T: (Double, Vector)](ReferenceFrame):
    """International Celestial Reference Frame"""

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
    def to_icrf(cls, q1: T, q2: T, q3: T) -> tuple[T, T, T]:
        return q1, q2, q3

    @classmethod
    def to_j2000(cls, q1: T, q2: T, q3: T) -> tuple[T, T, T]:
        """Convert from ICRF to J2000 reference frame"""
        q1, q2, q3 = np.dot(cls.B, np.array([q1, q2, q3]))
        return q1, q2, q3


class J2000[T: (Double, Vector)](ReferenceFrame):
    """Mean Equator and Equinox of J2000"""

    @classmethod
    def to_icrf(cls, q1: T, q2: T, q3: T) -> tuple[T, T, T]:
        """Convert from J2000 to ICRF reference frame"""
        q1, q2, q3 = np.dot(ICRF.B.T, np.array([q1, q2, q3]))
        return q1, q2, q3

    @classmethod
    def to_j2000(cls, q1: T, q2: T, q3: T) -> tuple[T, T, T]:
        return q1, q2, q3


class ITRF[T: (Double, Vector)](ReferenceFrame):
    """International Terrestrial Reference Frame"""

    variables = {
        "h": "q1",
        "lon": "q2",
        "lat": "q3",
        "dh": "q4",
        "dlon": "q5",
        "dlat": "q6",
    }

    @classmethod
    def to_cartesian(cls, q1: T, q2: T, q3: T) -> tuple[T, T, T]:
        """Conversion from geodetic spherical coordinates to cartesian"""

        # Earth flattening and squared eccentricity
        f = (Earth.Re - Earth.Rp) / Earth.Re
        e2 = 2.0 * f - f * f

        # Ecliptic and out-of-plane components
        sinq2 = np.sin(q2)
        C = Earth.Re / np.sqrt(1.0 - e2 * sinq2 * sinq2)
        S = C * (1.0 - e2)

        # Cartesian coordinates
        common = (C + q1) * np.cos(q2)
        cartesian_q1: Any = common * np.cos(q3)
        cartesian_q2: Any = common * np.sin(q3)
        cartesian_q3: Any = (S + q1) * sinq2

        return cartesian_q1, cartesian_q2, cartesian_q3

    @classmethod
    def to_spherical(cls, q1: T, q2: T, q3: T) -> tuple[T, T, T]:
        """Conversion from cartesian to geodetic spherical coordinates"""

        # Longitude and recurrent parameters
        sph_q3: Any = np.arctan2(q2, q1)
        f = (Earth.Re - Earth.Rp) / Earth.Re
        e2 = 2 * f - f * f
        p = np.sqrt(q1 * q1 + q2 * q2)

        # Iterate to find height and latitude
        sph_q2: Any = 0.0 * sph_q3
        dsph_q2 = sph_q2 + 1.0
        sph_q1 = sph_q2
        tol = 1e-15

        while np.any(np.abs(dsph_q2) > tol):
            shp_q2_old = sph_q2
            N = Earth.Re / np.sqrt(1.0 - e2 * np.sin(sph_q2) * np.sin(sph_q2))
            sph_q1 = p / np.cos(sph_q2) - N
            sph_q2 = np.arctan2(q3 * (N + sph_q1), p * (N + sph_q1 - e2 * N))
            dsph_q2 = sph_q2 - shp_q2_old

        return sph_q1, sph_q2, sph_q3


# class ReferenceFrame[T: (Double, Vector)]:
#     """Base class for reference frames"""

#     variables = {"r": "q1", "theta": "q2", "phi": "q3"}

#     @classmethod
#     def convert(cls, frame: type[Frame], *qs: T) -> tuple[T, ...]:
#         """Convert position vector to a different reference frame

#         :param r: Position vector in current frame
#         :param frame: Reference frame to convert to
#         """
#         if frame == ICRF:
#             return cls.__perform_conversion(cls.to_icrf, *qs)
#         elif frame == J2000:
#             return cls.__perform_conversion(cls.to_j2000, *qs)

#         raise NotImplementedError(
#             f"Conversion from {cls.__name__} to {frame.__name__} not implemented"
#         )

#     @classmethod
#     def __perform_conversion(cls, func, *qs: T) -> tuple[T, ...]:

#         if len(qs) == 3:
#             return func(*qs)
#         elif len(qs) == 6:
#             raise NotImplementedError("Conversion of velocity not implemented")
#         else:
#             raise ValueError("Invalid number of coordinates")

#     @classmethod
#     def to_icrf(cls, *qs: T) -> tuple[T, ...]:
#         raise NotImplementedError

#     @classmethod
#     def to_j2000(cls, *qs: T) -> tuple[T, ...]:
#         raise NotImplementedError

#     @classmethod
#     def to_gcrf(cls, *qs: T) -> tuple[T, ...]:
#         raise NotImplementedError

#     @classmethod
#     def to_spherical(cls, x: T, y: T, z: T) -> tuple[T, T, T]:
#         raise NotImplementedError

#     @classmethod
#     def to_cartesian(cls, r: T, theta: T, phi: T) -> tuple[T, T, T]:
#         raise NotImplementedError


# class ICRF[T: (Double, Vector)](ReferenceFrame):
#     """International Celestial Reference Frame

#     Inertial reference frame with origin at the barycenter of the solar system
#     and orientation defined from extragalactic sources. The ICRF is aligned with
#     the J2000 equatorial frame (Equator as XY plane and vernal equinox as X axis)
#     to within a few milliarcseconds.

#     Attributes
#     ----------
#     B : np.ndarray
#         Frame-bias matrix, used to convert from ICRF to J2000.

#     Sources
#     -------
#     https://arxiv.org/abs/astro-ph/0602086
#     """

#     variables = {"r": "q1", "alpha": "q2", "delta": "q3"}

#     dalpha0 = -14.6 * miliarcsec
#     xi0 = -16.6170 * miliarcsec
#     eta0 = -6.8192 * miliarcsec

#     B = np.array(
#         [
#             [
#                 1.0 - 0.5 * (dalpha0 * dalpha0 + xi0 * xi0),
#                 dalpha0,
#                 -xi0,
#             ],
#             [
#                 -dalpha0 - eta0 * xi0,
#                 1.0 - 0.5 * (dalpha0 * dalpha0 + eta0 * eta0),
#                 -eta0,
#             ],
#             [
#                 xi0 - eta0 * dalpha0,
#                 eta0 + xi0 * dalpha0,
#                 1.0 - 0.5 * (xi0 * xi0 + eta0 * eta0),
#             ],
#         ]
#     )

#     @classmethod
#     def to_icrf(cls, *qs: T) -> tuple[T, ...]:
#         return qs

#     @classmethod
#     def to_j2000(cls, *qs: T) -> tuple[T, ...]:
#         """Convert from ICRF to J2000 reference frame"""
#         out = np.dot(cls.B, np.array(qs))
#         return out[0], out[1], out[2]


# class J2000[T: (Double, Vector)](ReferenceFrame):
#     """Mean Equator and Equinox of J2000

#     Former standard reference frame for celestial coordinates, related to the
#     ICRF by the frame-bias matrix (B). The XY plane is the mean equator and the
#     X axis is the mean equinox of J2000.

#     Sources
#     -------
#     https://arxiv.org/abs/astro-ph/0602086
#     """

#     @classmethod
#     def to_j2000(cls, *qs: T) -> tuple[T, ...]:
#         return qs

#     @classmethod
#     def to_icrf(cls, *qs: T) -> tuple[T, ...]:
#         """Convert from J2000 to ICRF reference frame"""
#         out = np.dot(ICRF.B.T, np.array(qs))
#         return out[0], out[1], out[2]


# class GCRF[T: (Double, Vector)](ReferenceFrame):
#     """Geocentric Celestial Reference Frame

#     Geocentric version of ICRF
#     """

#     variables = {"r": "q1", "alpha": "q2", "delta": "q3"}


# class ITRF[T: (Double, Vector)](ReferenceFrame):
#     """International Terrestrial Reference Frame"""

#     variables = {"h": "q1", "lat": "q2", "lon": "q3"}

#     @classmethod
#     def to_cartesian(cls, r: T, theta: T, phi: T) -> tuple[T, T, T]:

#         height, lat, lon = r, theta, phi

#         # Compute flattening and eccentricity
#         f = (Earth.Re - Earth.Rp) / Earth.Re
#         e2 = 2 * f - f * f

#         # Compute ecliptic plane and out-of-plane components
#         sinlat = np.sin(lat)
#         C = Earth.Re / np.sqrt(1.0 - e2 * sinlat * sinlat)
#         S = C * (1.0 - e2)

#         # Compute Cartesian coordinates
#         rd = (C + height) * np.cos(lat)
#         x: Any = rd * np.cos(lon)
#         y: Any = rd * np.sin(lon)
#         z: Any = (S + height) * sinlat

#         return x, y, z

#     @classmethod
#     def to_spherical(cls, x: T, y: T, z: T) -> tuple[T, T, T]:

#         # Compute longitude and recurrent parameters
#         lon = np.arctan2(y, x)
#         f = (Earth.Re - Earth.Rp) / Earth.Re
#         e2 = 2 * f - f * f
#         p = np.sqrt(x * x + y * y)

#         # Iterate to find height and latitude
#         lat = 0.0 * lon
#         dlat = lat + 1.0
#         height = lat
#         tol = 1e-12

#         while np.any(np.abs(dlat) > tol):
#             lat0 = lat
#             N = Earth.Re / np.sqrt(1.0 - e2 * np.sin(lat) * np.sin(lat))
#             height = p / np.cos(lat) - N
#             lat = np.arctan2(z * (N + height), p * (N + height - e2 * N))
#             dlat = lat - lat0

#         height: Any = height
#         lat: Any = lat
#         lon: Any = lon

#         return height, lat, lon
