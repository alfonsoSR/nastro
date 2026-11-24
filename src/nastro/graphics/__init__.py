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

from .settings import PlotSetup
from .core import BaseFigure
from .figures import (
    SingleAxis,
    DoubleAxis,
    ParasiteAxis,
    Legend,
    Plot3D,
    Mosaic,
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
