"""
Types submodule
================

Fundamental types
------------------
===========  ==================  ====================================================
Type           Type Guard         Description
===========  ==================  ====================================================
``Scalar``    ``is_scalar``          A single integer or floating point number.
``Double``    ``is_double``          A single floating point number.
``Vector``    ``is_vector``          A 1D numpy array of floating point numbers.
``Array``     ``is_array``           A 1D numpy array or sequence of scalars.
===========  ==================  ====================================================

State representation
---------------------

.. autosummary::
    :toctree: generated
    :recursive:

    GenericState

"""

from .core import (
    Scalar,
    Double,
    Vector,
    Array,
    is_scalar,
    is_double,
    is_vector,
    is_array,
)

from .state import (
    GenericState,
    CartesianState,
    CartesianStateDerivative,
    KeplerianState,
    KeplerianStateDerivative,
    CartesianPosition,
    CartesianVelocity,
)

from .time import CalendarDate, JulianDay, UTC

from .conversion import time_to_mean_anomaly

__all__ = [
    # Core
    "Scalar",
    "Double",
    "Vector",
    "Array",
    "is_scalar",
    "is_double",
    "is_vector",
    "is_array",
    # State
    "GenericState",
    "CartesianState",
    "CartesianStateDerivative",
    "KeplerianState",
    "KeplerianStateDerivative",
    "CartesianPosition",
    "CartesianVelocity",
    # Time
    "CalendarDate",
    "JulianDay",
    "UTC",
    "time_to_mean_anomaly",
]
