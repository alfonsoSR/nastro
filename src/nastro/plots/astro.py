from ..types import KeplerianState, CartesianState, Vector
from ..constants import day
from . import core as ng

# TODO: ADD DOCSTRINGS


class StatePlot(ng.Mosaic):

    def __init__(self, setup: ng.PlotSetup = ng.PlotSetup()) -> None:

        super().__init__("ab;cd;ef", setup)

        base_setup = self.setup.copy()
        if self.setup.xlabel is None:
            base_setup.xlabel = "Days past initial epoch"

        subplot_setups = self.subplot_setup()

        self.q1_subplot = self.add_subplot(setup=subplot_setups[0])
        self.q2_subplot = self.add_subplot(setup=subplot_setups[1])
        self.q3_subplot = self.add_subplot(setup=subplot_setups[2])
        self.q4_subplot = self.add_subplot(setup=subplot_setups[3])
        self.q5_subplot = self.add_subplot(setup=subplot_setups[4])
        self.q6_subplot = self.add_subplot(setup=subplot_setups[5])

        return None

    def subplot_setup(self) -> list[ng.PlotSetup]:
        raise NotImplementedError

    def __exit__(self, exc_type, exc_value, traceback) -> None:

        self.q1_subplot.postprocess()
        self.q2_subplot.postprocess()
        self.q3_subplot.postprocess()
        self.q4_subplot.postprocess()
        self.q5_subplot.postprocess()
        self.q6_subplot.postprocess()

        super().__exit__(exc_type, exc_value, traceback)
        return None


class PlotState(StatePlot):

    def add_state(
        self,
        time: Vector,
        state: CartesianState | KeplerianState,
        fmt: str = "-",
        is_dt: bool = False,
    ) -> None:

        if not is_dt:
            time = (time - time[0]) / day

        self.q1_subplot.add_line(time, state.q1, fmt)
        self.q2_subplot.add_line(time, state.q2, fmt)

        if isinstance(state, KeplerianState):
            self.q3_subplot.add_line(time, state.i_deg, fmt)
            self.q4_subplot.add_line(time, state.raan_deg, fmt)
            self.q5_subplot.add_line(time, state.aop_deg, fmt)
            self.q6_subplot.add_line(time, state.ta_deg, fmt)
        else:
            self.q3_subplot.add_line(time, state.q3, fmt)
            self.q4_subplot.add_line(time, state.q4, fmt)
            self.q5_subplot.add_line(time, state.q5, fmt)
            self.q6_subplot.add_line(time, state.q6, fmt)

        return None


class CompareState(StatePlot):

    def compare_states[
        T: (CartesianState, KeplerianState)
    ](
        self,
        time: Vector,
        orbit: T,
        reference: T,
        is_dt: bool = False,
        fmt: str = "-",
    ) -> None:

        if not is_dt:
            time = (time - time[0]) / day

        ds = orbit - reference
        self.q1_subplot.add_line(time, ds.q1, fmt)
        self.q2_subplot.add_line(time, ds.q2, fmt)
        self.q3_subplot.add_line(time, ds.q3, fmt)
        self.q4_subplot.add_line(time, ds.q4, fmt)
        self.q5_subplot.add_line(time, ds.q5, fmt)
        self.q6_subplot.add_line(time, ds.q6, fmt)

        return None


class PlotKeplerianState(PlotState):
    """Plot components of a keplerian state vector"""

    def subplot_setup(self) -> list[ng.PlotSetup]:

        base_setup = self.setup.copy()

        a_setup = base_setup.copy()
        a_setup.ylabel = r"$a\ [m]$"

        e_setup = base_setup.copy()
        e_setup.ylabel = r"$e$"

        i_setup = base_setup.copy()
        i_setup.ylabel = r"$i\ [deg]$"

        aop_setup = base_setup.copy()
        aop_setup.ylabel = r"$\omega\ [deg]$"

        raan_setup = base_setup.copy()
        raan_setup.ylabel = r"$\Omega\ [deg]$"

        ta_setup = base_setup.copy()
        ta_setup.ylabel = r"$\theta\ [deg]$"

        return [a_setup, e_setup, i_setup, aop_setup, raan_setup, ta_setup]


class PlotCartesianState(PlotState):
    """Plot components of a cartesian state vector

    Parameters
    -----------
    setup : PlotSetup
        Plot setup
    """

    def subplot_setup(self) -> list[ng.PlotSetup]:

        base_setup = self.setup.copy()

        x_setup = base_setup.copy()
        x_setup.ylabel = r"$x\ [m]$"

        y_setup = base_setup.copy()
        y_setup.ylabel = r"$y\ [m]$"

        z_setup = base_setup.copy()
        z_setup.ylabel = r"$z\ [m]$"

        vx_setup = base_setup.copy()
        vx_setup.ylabel = r"$\dot x\ [m/s]$"

        vy_setup = base_setup.copy()
        vy_setup.ylabel = r"$\dot y\ [m/s]$"

        vz_setup = base_setup.copy()
        vz_setup.ylabel = r"$\dot z\ [m/s]$"

        return [x_setup, y_setup, z_setup, vx_setup, vy_setup, vz_setup]


class CompareCartesianStates(CompareState):
    """Plot difference between two sets of cartesian states"""

    def subplot_setup(self) -> list[ng.PlotSetup]:

        base_setup = self.setup.copy()

        dx_setup = base_setup.copy()
        dx_setup.ylabel = r"$\Delta x\ [m]$"

        dy_setup = base_setup.copy()
        dy_setup.ylabel = r"$\Delta y\ [m]$"

        dz_setup = base_setup.copy()
        dz_setup.ylabel = r"$\Delta z\ [m]$"

        dvx_setup = base_setup.copy()
        dvx_setup.ylabel = r"$\Delta \dot x\ [m/s]$"

        dvy_setup = base_setup.copy()
        dvy_setup.ylabel = r"$\Delta \dot y\ [m/s]$"

        dvz_setup = base_setup.copy()
        dvz_setup.ylabel = r"$\Delta \dot z\ [m/s]$"

        return [dx_setup, dy_setup, dz_setup, dvx_setup, dvy_setup, dvz_setup]


class CompareKeplerianStates(CompareState):
    """Plot difference between two sets of keplerian states"""

    def subplot_setup(self) -> list[ng.PlotSetup]:

        base_setup = self.setup.copy()

        da_setup = base_setup.copy()
        da_setup.ylabel = r"$\Delta a\ [m]$"

        de_setup = base_setup.copy()
        de_setup.ylabel = r"$\Delta e$"

        di_setup = base_setup.copy()
        di_setup.ylabel = r"$\Delta i\ [rad]$"

        daop_setup = base_setup.copy()
        daop_setup.ylabel = r"$\Delta \omega\ [rad]$"

        draan_setup = base_setup.copy()
        draan_setup.ylabel = r"$\Delta \Omega\ [rad]$"

        dta_setup = base_setup.copy()
        dta_setup.ylabel = r"$\Delta \theta\ [rad]$"

        return [da_setup, de_setup, di_setup, daop_setup, draan_setup, dta_setup]


class PlotOrbit(ng.Base3D):

    def add_orbit(
        self,
        state: CartesianState,
        fmt: str = "-",
        width: float | None = None,
        markersize: float | None = None,
        color: str | None = None,
        alpha: float = 1.0,
        label: str | None = None,
        axis: str = "left",
    ) -> None:

        self.add_line(
            state.x, state.y, state.z, fmt, width, markersize, color, alpha, label, axis
        )

        return None
