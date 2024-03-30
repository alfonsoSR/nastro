from dataclasses import dataclass


@dataclass(frozen=True)
class CelestialBody:
    """Properties of celestial bodies

    Sources:
        - GM data: DE440 ephemeris
        - Earth data: Horizons geophysical data [March 2024]

    :param R: Mean radius [m]
    :param mu: Gravitational parameter [m^3/s^2]
    :param a: Reference semi-major axis [m]
    :param e: Reference eccentricity
    :param i: Reference inclination [deg]
    :param omega: Reference argument of periapsis [deg]
    :param Omega: Reference right ascension of the ascending node [deg]
    :param j2: Second zonal harmonic
    :param j3: Third zonal harmonic
    :param T: Orbital period [s] (Calculated)
    """

    name: str
    _R: float | None = None
    _mu: float | None = None
    _j2: float | None = None
    _j3: float | None = None
    _T: float | None = None
    _a: float | None = None

    @property
    def R(self) -> float:
        if self._R is None:
            raise ValueError(f"Mean radius is not defined for {self.name}")
        return self._R

    @property
    def mu(self) -> float:
        if self._mu is None:
            raise ValueError(f"Gravitational parameter is not defined for {self.name}")
        return self._mu

    @property
    def j2(self) -> float:
        if self._j2 is None:
            raise ValueError(f"Second zonal harmonic is not specified for {self.name}")
        return self._j2

    @property
    def j3(self) -> float:
        if self._j3 is None:
            raise ValueError(f"Third zonal harmonic is not specified for {self.name}")
        return self._j3

    @property
    def T(self) -> float:
        if self._T is None:
            raise ValueError(f"Orbital period is not specified for {self.name}")
        return self._T

    @property
    def a(self) -> float:
        if self._a is None:
            raise ValueError(
                f"Reference semi-major axis is not specified for {self.name}"
            )
        return self._a

    @property
    def omega(self) -> float:
        return 1.0 / self.T


Sun = CelestialBody(
    "Sun",
    _mu=1.3271244004127942e20,
)

Mercury = CelestialBody(
    "Mercury",
    _R=None,
    _mu=2.2031868551400003e13,
)

Venus = CelestialBody(
    "Venus",
    _R=None,
    _mu=3.2485859200000000e14,
    _a=108.210e9,
)

Earth = CelestialBody(
    "Earth",
    _R=6.37101e6,
    _mu=3.9860043550702266e14,
    _j2=1.08262545e-3,
    _T=3.155814950400000e07,
    _a=149.6e9,
)

Mars = CelestialBody(
    "Mars",
    _R=None,
    _mu=4.282837362069909e13,
)

Jupiter = CelestialBody(
    "Jupiter",
    _R=71492.0e3,
    _mu=1.266865341960128e17,
    _j2=14.696572e-3,
    _j3=-0.042e-6,
)

Saturn = CelestialBody(
    "Saturn",
    _R=None,
    _mu=3.793120623436167e16,
)

Uranus = CelestialBody(
    "Uranus",
    _R=None,
    _mu=5.793951256527211e15,
)

Neptune = CelestialBody(
    "Neptune",
    _R=None,
    _mu=6.835103145462294e15,
)

Pluto = CelestialBody(
    "Pluto",
    _R=None,
    _mu=8.696138177608748e11,
)

# Moons
Moon = CelestialBody(
    "Moon",
    _R=1.7374e6,
    _mu=4.9028001184575496e12,
)

Phobos = CelestialBody(
    "Phobos",
    _R=None,
    _mu=7.087546066894452e5,
)

Deimos = CelestialBody(
    "Deimos",
    _R=None,
    _mu=9.615569648120313e4,
)

Io = CelestialBody(
    "Io",
    _R=1821.6e3,
    _mu=5.959924010272514e12,
    _T=1.5292800000000e05,
)

Europa = CelestialBody(
    "Europa",
    _R=1565.0e3,
    _mu=3.202739815114734e12,
    _T=3.0672000000000e05,
)

Ganymede = CelestialBody(
    "Ganymede",
    _R=2634.0e3,
    _mu=9.887819980080976e12,
    _T=6.1819200000000e05,
)

Callisto = CelestialBody(
    "Callisto",
    _R=2410.3e3,
    _mu=7.179304867611079e12,
    _T=1.4421024000000e06,
)

Mimas = CelestialBody(
    "Mimas",
    _R=None,
    _mu=2.503488768152587e9,
)

Enceladus = CelestialBody(
    "Enceladus",
    _R=None,
    _mu=7.210366688598896e9,
)

Tethys = CelestialBody(
    "Tethys",
    _R=None,
    _mu=4.121352885489587e10,
)

Dione = CelestialBody(
    "Dione",
    _R=None,
    _mu=7.311607172482067e10,
)

Rhea = CelestialBody(
    "Rhea",
    _R=None,
    _mu=1.539417519146563e11,
)

Titan = CelestialBody(
    "Titan",
    _R=None,
    _mu=8.978137095521046e12,
)

Ariel = CelestialBody(
    "Ariel",
    _R=None,
    _mu=8.346344431770477e10,
)

Umbriel = CelestialBody(
    "Umbriel",
    _R=None,
    _mu=8.509338094489388e10,
)

Titania = CelestialBody(
    "Titania",
    _R=None,
    _mu=2.269437003741248e11,
)

Oberon = CelestialBody(
    "Oberon",
    _R=None,
    _mu=2.053234302535623e11,
)

Miranda = CelestialBody(
    "Miranda",
    _R=None,
    _mu=4.319516899232100e9,
)

Triton = CelestialBody(
    "Triton",
    _R=None,
    _mu=1.428495462910464e12,
)

Charon = CelestialBody(
    "Charon",
    _R=None,
    _mu=1.058799888601881e11,
)
