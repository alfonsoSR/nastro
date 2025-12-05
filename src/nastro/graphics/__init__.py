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

from .core import (
    PlotSetup,
    Mosaic,
    SingleAxis,
    DoubleAxis,
    ParasiteAxis,
    Plot3D,
    BaseFigure,
    Legend,
)
from .astro import (
    PlotCartesianState,
    PlotKeplerianState,
    CompareCartesianStates,
    CompareKeplerianStates,
    CompareRswStates,
    PlotOrbit,
)
from . import shapes

__all__ = [
    "BaseFigure",
    "PlotSetup",
    "Mosaic",
    "SingleAxis",
    "DoubleAxis",
    "ParasiteAxis",
    "Legend",
    "Plot3D",
    "PlotCartesianState",
    "PlotKeplerianState",
    "CompareCartesianStates",
    "CompareKeplerianStates",
    "CompareRswStates",
    "PlotOrbit",
    "shapes",
]
