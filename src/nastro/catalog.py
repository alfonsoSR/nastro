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
from . import constants as nc


class CelestialBody:
    """Properties of celestial bodies

    Sources:
    - GM data for planets: https://doi.org/10.3847/1538-3881/abd414
    - Radii: https://doi.org/10.1007/s10569-017-9805-5
    - Earth data: Horizons geophysical data [March 2024]
    - Physical properties of the Moon: https://doi.org/10.1007/s11214-019-0613-y

    Properties
    ----------
    name : str
        Name of the celestial body
    R : Double
        Mean radius [m]
    Rp : Double
        Polar radius [m]
    Re : Double
        Equatorial radius [m]
    rho : Double
        Mean density [kg/m^3]
    mu : Double
        Gravitational parameter [m^3/s^2]
    j2 : Double
        Second zonal harmonic
    j3 : Double
        Third zonal harmonic
    T : Double
        Orbital period [s]
    a : Double
        Semi-major axis [m]
    mass : Double
        Mass [kg]
    volume : Double
        Volume [m^3]
    inertia_factor : Double
        Inertia factor (I/MR^2)
    """

    _attributes = [
        "R",
        "Rp",
        "Re",
        "mu",
        "j2",
        "j3",
        "T",
        "a",
        "mass",
        "volume",
        "rho",
        "inertia_factor",
    ]

    def __init__(
        self,
        name: str,
        R: Double | None = None,
        Rp: Double | None = None,
        Re: Double | None = None,
        mu: Double | None = None,
        j2: Double | None = None,
        j3: Double | None = None,
        T: Double | None = None,
        a: Double | None = None,
        inertia_factor: Double | None = None,
    ) -> None:
        self.name = name
        self.__R = R
        self.__Rp = Rp
        self.__Re = Re
        self.__mu = mu
        self.__j2 = j2
        self.__j3 = j3
        self.__T = T
        self.__a = a
        self.__inertia_factor = inertia_factor

        self.__mass = None if mu is None else mu / nc.G
        self.__volume = None if R is None else (4.0 / 3.0) * nc.pi * (R**3)
        self.__rho = (
            None
            if self.__mass is None or self.__volume is None
            else self.__mass / self.__volume
        )

    def __getattr__(self, name: str) -> Double:

        if name in self._attributes:

            val = self.__getattribute__(f"_CelestialBody__{name}")
            if val is None:
                raise AttributeError(f"{name} is not specified for '{self.name}'")
            return val

        raise AttributeError(f"'{self.name}' has no attribute '{name}'")

    def __repr__(self) -> str:

        out = f"Physical properties: {self.name}\n"
        out += "-" * len(out) + "\n"
        for attr in self._attributes:
            try:
                val = getattr(self, attr)
                out += f"{attr}:\t{val:e}\n"
            except AttributeError:
                continue
        return out


Sun = CelestialBody(
    "Sun",
    mu=1.3271244004127942e20,
    Re=695700e3,
)

Mercury = CelestialBody(
    "Mercury",
    R=2.4394e6,
    Re=2440.53e3,
    Rp=2438.26e3,
    mu=2.2031868551400003e13,
)

Venus = CelestialBody(
    "Venus",
    R=6051.8e3,
    Re=6051.8e3,
    Rp=6051.8e3,
    mu=3.2485859200000000e14,
    a=108.210e9,
)

Earth = CelestialBody(
    "Earth",
    R=6371.0084e3,
    Re=6378.1366e3,
    Rp=6356.7519e3,
    mu=3.9860043550702266e14,
    j2=1.08262545e-3,
    T=3.155814950400000e07,
    a=149.6e9,
)

Mars = CelestialBody(
    "Mars",
    R=3389.50e3,
    Re=3396.19e3,
    mu=4.282837362069909e13,
)

Jupiter = CelestialBody(
    "Jupiter",
    R=69911e3,
    Re=71492e3,
    Rp=66854e3,
    mu=1.266865341960128e17,
    j2=14.696572e-3,
    j3=-0.042e-6,
)

Saturn = CelestialBody(
    "Saturn",
    R=58232e3,
    Re=60268e3,
    Rp=54364e3,
    mu=3.793120623436167e16,
)

Uranus = CelestialBody(
    "Uranus",
    R=25362e3,
    Re=25559e3,
    Rp=24973e3,
    mu=5.793951256527211e15,
)

Neptune = CelestialBody(
    "Neptune",
    R=24622e3,
    Re=24764e3,
    Rp=24341e3,
    mu=6.835103145462294e15,
)


# Moons
Moon = CelestialBody(
    "Moon",
    R=1.7374e6,
    mu=4.9028001184575496e12,
    inertia_factor=0.3931,
)

Phobos = CelestialBody(
    "Phobos",
    R=11.08e3,
    mu=7.087546066894452e5,
)

Deimos = CelestialBody(
    "Deimos",
    R=6.2e3,
    mu=9.615569648120313e4,
)

Io = CelestialBody(
    "Io",
    R=1821.49e3,
    mu=5.959924010272514e12,
    T=1.5292800000000e05,
)

Europa = CelestialBody(
    "Europa",
    R=1560.8e3,
    mu=3.202739815114734e12,
    T=3.0672000000000e05,
)

Ganymede = CelestialBody(
    "Ganymede",
    R=2631.2e3,
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
    R=198.2e3,
    mu=2.503488768152587e9,
)

Enceladus = CelestialBody(
    "Enceladus",
    R=252.1e3,
    mu=7.210366688598896e9,
)

Tethys = CelestialBody(
    "Tethys",
    R=531.0e3,
    mu=4.121352885489587e10,
)

Dione = CelestialBody(
    "Dione",
    R=561.4e3,
    mu=7.311607172482067e10,
)

Rhea = CelestialBody(
    "Rhea",
    R=763.5e3,
    mu=1.539417519146563e11,
)

Titan = CelestialBody(
    "Titan",
    R=2575.0e3,
    mu=8.978137095521046e12,
)

Ariel = CelestialBody(
    "Ariel",
    R=578.9e3,
    mu=8.346344431770477e10,
)

Umbriel = CelestialBody(
    "Umbriel",
    R=584.7e3,
    mu=8.509338094489388e10,
)

Titania = CelestialBody(
    "Titania",
    R=788.9e3,
    mu=2.269437003741248e11,
)

Oberon = CelestialBody(
    "Oberon",
    R=761.4e3,
    mu=2.053234302535623e11,
)

Miranda = CelestialBody(
    "Miranda",
    R=235.8e3,
    mu=4.319516899232100e9,
)

Triton = CelestialBody(
    "Triton",
    R=1352.6e3,
    mu=1.428495462910464e12,
)

# Dwarf planets
Pluto = CelestialBody(
    "Pluto",
    R=1188.3e3,
    mu=8.696138177608748e11,
)

Charon = CelestialBody(
    "Charon",
    R=606.0e3,
    mu=1.058799888601881e11,
)

Ceres = CelestialBody(
    "Ceres",
    R=470e3,
)
