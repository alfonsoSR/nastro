"""
Types submodule
================

.. currentmodule:: nastro.types

Type aliases
-------------

=============== ==============================================================
`Double`        Double precision floating point number
`Vector`        One-dimensional array of double precision floating point numbers
`ArrayLike`     Sequence of elements that can be converted to a numpy array
=============== ==============================================================

State definition
-----------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

    GenericState
    CartesianState
    CartesianStateDerivative
    CartesianPosition
    CartesianVelocity
    KeplerianState
    KeplerianStateDerivative

Time
-----

.. autosummary::
    :toctree: generated/
    :nosignatures:

    JulianDay
    CalendarDate
    UTC

"""

from .core import Double, Vector, ArrayLike

from .state import (
    GenericState,
    CartesianState,
    CartesianStateDerivative,
    KeplerianState,
    KeplerianStateDerivative,
    CartesianPosition,
    CartesianVelocity,
    SphericalPosition,
)

from .time import CalendarDate, JulianDay, UTC

from .conversion import time_to_mean_anomaly

__all__ = [
    "Double",
    "Vector",
    "ArrayLike",
    "GenericState",
    "CartesianState",
    "CartesianStateDerivative",
    "KeplerianState",
    "KeplerianStateDerivative",
    "CartesianPosition",
    "CartesianVelocity",
    "SphericalPosition",
    "CalendarDate",
    "JulianDay",
    "UTC",
    "time_to_mean_anomaly",
]
