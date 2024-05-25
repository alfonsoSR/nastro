"""
Plotting submodule
==================

Generic plots
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    PlotSetup
    SingleAxis
    DoubleAxis
    ParasiteAxis
    Mosaic


Astrodynamics plots
--------------------

.. autosummary::
    :toctree: generated/
    :nosignatures:
    :recursive:

    PlotCartesianState
    PlotKeplerianState
    CompareCartesianStates
    CompareKeplerianStates

"""

from .core import PlotSetup, Mosaic, SingleAxis, DoubleAxis, ParasiteAxis, Plot3D
from .astro import (
    PlotCartesianState,
    PlotKeplerianState,
    CompareCartesianStates,
    CompareKeplerianStates,
    PlotOrbit,
)

__all__ = [
    "PlotSetup",
    "Mosaic",
    "SingleAxis",
    "DoubleAxis",
    "ParasiteAxis",
    "Plot3D",
    "PlotCartesianState",
    "PlotKeplerianState",
    "CompareCartesianStates",
    "CompareKeplerianStates",
    "PlotOrbit",
]
