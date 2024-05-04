from nastro import types as nt, plots as ng
from pathlib import Path

if __name__ == "__main__":

    sourcedir = Path(__file__).parents[1] / "data"
    cstate = nt.CartesianState.load(sourcedir / "cstate.npy")

    with ng.PlotOrbit() as figure:

        figure.add_orbit(cstate, color="green", label="Orbit")
