from nastro.plots.astro import (
    PlotKeplerianState,
    PlotCartesianState,
    CompareCartesianStates,
    CompareKeplerianStates,
)
import nastro.plots as pp
from nastro.types import Vector, KeplerianState, JulianDate
import numpy as np
from pathlib import Path
from nastro.catalog import Ganymede

if __name__ == "__main__":

    datadir = Path(__file__).parents[1] / "data"
    time = JulianDate.load(datadir / "epochs.npy").jd[:600]
    kstates = KeplerianState.load(datadir / "kstate.npy")[:600]
    kref = KeplerianState(*kstates[0].asarray())
    setup = pp.PlotSetup(grid=True)
    cstates = kstates.as_cartesian(Ganymede.mu)
    cref = kref.as_cartesian(Ganymede.mu)

    with PlotKeplerianState(setup) as fig:
        fig.add_state(time, kstates, is_dt=False)

    with PlotCartesianState(setup) as fig:
        fig.add_state(time, cstates, is_dt=False)

    with CompareCartesianStates(setup) as fig:
        fig.compare_states(time, cstates, cref, is_dt=False)

    with CompareKeplerianStates(setup) as fig:
        fig.compare_states(time, kstates, kref, is_dt=False, fmt=".")
