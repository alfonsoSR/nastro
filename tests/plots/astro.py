from nastro import graphics as ng, types as nt, catalog as ncat
import numpy as np
from pathlib import Path

if __name__ == "__main__":

    datadir = Path(__file__).parents[1] / "data"
    time = nt.JulianDay.load(datadir / "epochs.npy").jd[:600]
    kstates = nt.KeplerianState.load(datadir / "kstate.npy")[:600]
    kref = nt.KeplerianState(*kstates[0].asarray)
    setup = ng.PlotSetup(grid=True)
    cstates = kstates.as_cartesian(ncat.Ganymede.mu)
    cref = kref.as_cartesian(ncat.Ganymede.mu)

    with ng.PlotKeplerianState(setup) as fig:
        fig.add_state(time, kstates, is_dt=False)

    with ng.PlotCartesianState(setup) as fig:
        fig.add_state(time, cstates, is_dt=False)

    with ng.CompareCartesianStates(setup) as fig:
        fig.compare_states(time, cstates, cref, is_dt=False)

    with ng.CompareKeplerianStates(setup) as fig:
        fig.compare_states(time, kstates, kref, is_dt=False, fmt=".")
