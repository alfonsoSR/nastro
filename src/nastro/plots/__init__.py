"""
Plotting submodule
==================

.. currentmodule:: nastro.plots

Generic plots
--------------

.. autosummary::
    :toctree: generated/
    :nosignatures:

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
