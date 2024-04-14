"""
Catalog
==============

.. module:: nastro.catalog

Base class
----------------
.. autosummary::
    :toctree: generated/
    :nosignatures:

    CelestialBody

Celestial bodies
------------------

=============================== ================================================
Planet                          Moons
=============================== ================================================
Sun                             None
Mercury                         None
Venus                           None
Earth                           Moon
Mars                            Phobos, Deimos
Jupiter                         Io, Europa, Ganymede, Callisto
Saturn                          Mimas, Enceladus, Tethys, Dione, Rhea, Titan
Uranus                          Ariel, Umbriel, Titania, Oberon, Miranda
Neptune                         Triton
Pluto                           Charon
=============================== ================================================
"""

from .types.core import Double


class CelestialBody:
    """Properties of celestial bodies

    Sources:
        - GM data: DE440 ephemeris
        - Earth data: Horizons geophysical data [March 2024]

    Parameters
    ----------
    name : str
        Name of the celestial body
    R : Double, optional
        Radius [m]
    mu : Double, optional
        Gravitational parameter [m^3/s^2]
    j2 : Double, optional
        Second zonal harmonic
    j3 : Double, optional
        Third zonal harmonic
    T : Double, optional
        Orbital period [s]
    a : Double, optional
        Semi-major axis [m]
    """

    def __init__(
        self,
        name: str,
        R: Double | None = None,
        mu: Double | None = None,
        j2: Double | None = None,
        j3: Double | None = None,
        T: Double | None = None,
        a: Double | None = None,
    ) -> None:
        self.name = name
        self.__R = R
        self.__mu = mu
        self.__j2 = j2
        self.__j3 = j3
        self.__T = T
        self.__a = a

    def __getattr__(self, name: str) -> Double:

        if name in ["R", "mu", "j2", "j3", "T", "a"]:

            val = self.__getattribute__(f"_CelestialBody__{name}")
            if val is None:
                raise AttributeError(f"{name} is not specified for '{self.name}'")
            return val

        raise AttributeError(f"'{self.name}' has no attribute '{name}'")


Sun = CelestialBody(
    "Sun",
    mu=1.3271244004127942e20,
)

Mercury = CelestialBody(
    "Mercury",
    R=None,
    mu=2.2031868551400003e13,
)

Venus = CelestialBody(
    "Venus",
    R=None,
    mu=3.2485859200000000e14,
    a=108.210e9,
)

Earth = CelestialBody(
    "Earth",
    R=6.37101e6,
    mu=3.9860043550702266e14,
    j2=1.08262545e-3,
    T=3.155814950400000e07,
    a=149.6e9,
)

Mars = CelestialBody(
    "Mars",
    R=None,
    mu=4.282837362069909e13,
)

Jupiter = CelestialBody(
    "Jupiter",
    R=71492.0e3,
    mu=1.266865341960128e17,
    j2=14.696572e-3,
    j3=-0.042e-6,
)

Saturn = CelestialBody(
    "Saturn",
    R=None,
    mu=3.793120623436167e16,
)

Uranus = CelestialBody(
    "Uranus",
    R=None,
    mu=5.793951256527211e15,
)

Neptune = CelestialBody(
    "Neptune",
    R=None,
    mu=6.835103145462294e15,
)

Pluto = CelestialBody(
    "Pluto",
    R=None,
    mu=8.696138177608748e11,
)

# Moons
Moon = CelestialBody(
    "Moon",
    R=1.7374e6,
    mu=4.9028001184575496e12,
)

Phobos = CelestialBody(
    "Phobos",
    R=None,
    mu=7.087546066894452e5,
)

Deimos = CelestialBody(
    "Deimos",
    R=None,
    mu=9.615569648120313e4,
)

Io = CelestialBody(
    "Io",
    R=1821.6e3,
    mu=5.959924010272514e12,
    T=1.5292800000000e05,
)

Europa = CelestialBody(
    "Europa",
    R=1565.0e3,
    mu=3.202739815114734e12,
    T=3.0672000000000e05,
)

Ganymede = CelestialBody(
    "Ganymede",
    R=2634.0e3,
    mu=9.887819980080976e12,
    T=6.1819200000000e05,
)

Callisto = CelestialBody(
    "Callisto",
    R=2410.3e3,
    mu=7.179304867611079e12,
    T=1.4421024000000e06,
)

Mimas = CelestialBody(
    "Mimas",
    R=None,
    mu=2.503488768152587e9,
)

Enceladus = CelestialBody(
    "Enceladus",
    R=None,
    mu=7.210366688598896e9,
)

Tethys = CelestialBody(
    "Tethys",
    R=None,
    mu=4.121352885489587e10,
)

Dione = CelestialBody(
    "Dione",
    R=None,
    mu=7.311607172482067e10,
)

Rhea = CelestialBody(
    "Rhea",
    R=None,
    mu=1.539417519146563e11,
)

Titan = CelestialBody(
    "Titan",
    R=None,
    mu=8.978137095521046e12,
)

Ariel = CelestialBody(
    "Ariel",
    R=None,
    mu=8.346344431770477e10,
)

Umbriel = CelestialBody(
    "Umbriel",
    R=None,
    mu=8.509338094489388e10,
)

Titania = CelestialBody(
    "Titania",
    R=None,
    mu=2.269437003741248e11,
)

Oberon = CelestialBody(
    "Oberon",
    R=None,
    mu=2.053234302535623e11,
)

Miranda = CelestialBody(
    "Miranda",
    R=None,
    mu=4.319516899232100e9,
)

Triton = CelestialBody(
    "Triton",
    R=None,
    mu=1.428495462910464e12,
)

Charon = CelestialBody(
    "Charon",
    R=None,
    mu=1.058799888601881e11,
)
