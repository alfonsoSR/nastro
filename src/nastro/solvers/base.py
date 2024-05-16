from ..forces import ForceModel
from .. import types as nt, constants as nc
import numpy as np
from typing import Any
from scipy.integrate import solve_ivp


class SimplePropagator:

    def __init__(self, model: ForceModel) -> None:

        self.model = model

        return None

    def fun(self, t: nt.Double, s: nt.Vector) -> nt.Vector:
        return self.model(t, nt.CartesianState(*s)).asarray.ravel()

    def propagate_scipy(
        self,
        t0: nt.Double,
        tend: nt.Double,
        s0: nt.CartesianState,
        solver: str = "RK45",
        atol: nt.Double = 1e-10,
        rtol: nt.Double = 1e-10,
    ) -> tuple[nt.Vector, nt.CartesianState]:

        sol = solve_ivp(
            self.fun,
            (t0, tend),
            s0.asarray.ravel(),
            method=solver,
            atol=atol,
            rtol=rtol,
        )

        return sol.t, nt.CartesianState(*sol.y)


class SimpleCartesianPropagator:

    def __init__(
        self,
        model: ForceModel,
        t0: nt.Double,
        s0: nt.CartesianState,
        tend: nt.Double,
        dt: nt.Double,
    ) -> None:

        self.model = model
        self.now = t0
        self.tend = tend
        if not s0.scalar:
            raise TypeError("Initial state must be scalar")
        self.s = s0
        self.h = dt

        # Termination criteria
        EPS = 1e-14
        self.done = max(EPS, 0.1 * dt)
        self.status = "running"

        return None

    def fun(self, t: nt.Double, s: nt.Vector) -> nt.Vector:
        return self.model(t, nt.CartesianState(*s)).asarray.ravel()

    def scipy_propagate(
        self, solver: str = "RK45", atol: nt.Double = 1e-12, rtol: nt.Double = 1e-12
    ) -> tuple[nt.Vector, nt.CartesianState]:

        sol = solve_ivp(
            self.fun,
            (self.now, self.tend),
            self.s.asarray.ravel(),
            method=solver,
            atol=atol,
            rtol=rtol,
        )

        return sol.t, nt.CartesianState(*sol.y)

    def propagate(self) -> tuple[nt.Vector, nt.CartesianState]:

        out_state = nt.CartesianState(*self.s.asarray)
        out_time = [self.now]

        # Main loop
        status = None
        while status is None:

            # Perform integration step
            self.step()

            # Check status after step
            if self.status == "finished":
                status = 0
            elif self.status == "failed":
                status = 1
                break

            # Update output
            out_time.append(self.now)
            out_state.append(self.s)

        return np.array(out_time), out_state

    def step(self) -> None:

        # Check status before performing step
        if self.status != "running":
            raise RuntimeError("Attempted step on non-running propagator")

        # Perform step
        success = self._step_impl()

        # Check for termination condition
        if not success:
            self.status = "failed"
        else:
            if self.tend - self.now < self.done:
                self.status = "finished"

        return None

    def _step_impl(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method")


class CartesianPropagator[U: (nt.Double, nt.JulianDay)]:
    """Base class for cartesian propagators

    Parameters
    -----------
    model : ForceModel
        Dynamical model representing the accelerations acting on the body
    t0, tend : Double | JulianDay[Double]
        Initial and final epochs
    s0 : CartesianState[Double]
        Initial state
    dt : Double
        Time step. Assumed to be in seconds if t0 and tend are Doubles and days
        if they are JulianDays
    """

    def __init__(
        self,
        model: ForceModel,
        t0: U,
        s0: nt.CartesianState[nt.Double],
        tend: U,
        dt: nt.Double,
    ) -> None:

        self.model = model

        # Initial and final epoch and step
        if isinstance(t0, nt.JulianDay) and isinstance(tend, nt.JulianDay):
            self.is_day = True
            self.now_int, self.now_fr = t0.day, t0.time
            self.tend_int, self.tend_fr = tend.day, tend.time
            dt /= nc.day
        elif nt.is_double(t0) and nt.is_double(tend):
            self.is_day = False
            self.now_int, self.now_fr = t0, 0.0
            self.tend_int, self.tend_fr = tend, 0.0
        else:
            raise TypeError("Unexpected type for initial or final epoch")

        self.now = self.now_int + self.now_fr
        self.tend = self.tend_int + self.tend_fr

        # Initial state
        if not s0.scalar:
            raise TypeError("Initial state must be scalar")
        self.s = s0

        # Step size
        delta_int = self.tend_int - self.now_int
        delta_fr = self.tend_fr - self.now_fr
        steps = (delta_int // dt) + (delta_fr // dt)
        self.h = (delta_int / steps) + (delta_fr / steps)
        self.h_int = np.trunc(self.h)
        self.h_fr = self.h - self.h_int

        if self.is_day:
            self.h_sec = self.h_int * nc.day + self.h_fr * nc.day
        else:
            self.h_sec = self.h_int + self.h_fr

        # Termination criteria
        self.EPS = 1e-14
        self.done_int = max(self.EPS, 0.1 * self.h_int)
        self.done_fr = max(self.EPS, 0.1 * self.h_fr)

        # Status
        self.status = "running"

        return None

    def _update_epoch(self) -> tuple[nt.Double, nt.Double]:

        new_fr = self.now_fr + self.h_fr
        int_part = np.trunc(new_fr)
        new_fr -= int_part
        new_int = self.now_int + self.h_int + int_part

        return new_int, new_fr

    def _get_steps(self) -> tuple[nt.Double, nt.Double]:

        if self.is_day:
            return self.h_int * nc.day, self.h_fr * nc.day
        else:
            return self.h_int, self.h_fr

    def propagate(self, frac: bool = True) -> tuple[U, nt.CartesianState]:
        """Propagate initial state"""

        out_sol = nt.CartesianState(*self.s.asarray)
        out_time = [self.now_int]
        out_time_fr = [self.now_fr]

        # Main loop
        prop_status = None
        while prop_status is None:

            # Perform integration step
            self.step()

            # Check status after step
            if self.status == "finished":
                prop_status = 0
            elif self.status == "failed":
                prop_status = 1
                break

            # Update state and time
            out_sol.append(self.s)
            out_time.append(self.now_int)
            out_time_fr.append(self.now_fr)

        # Generate output
        time_int = np.array(out_time, dtype=np.float64)
        time_fr = np.array(out_time_fr, dtype=np.float64)
        time: Any = (
            nt.JulianDay(time_int, time_fr) if self.is_day else time_int + time_fr
        )

        return time, out_sol

    def step(self) -> None:
        """Perform a single integration step"""

        # Check status before performing step
        if self.status != "running":
            raise RuntimeError("Attempted step on non-running propagator")

        # Perform step
        success = self._step_impl()

        # Check for termination condition
        if not success:
            self.status = "failed"
        else:
            if (self.tend_int - self.now_int < self.done_int) and (
                self.tend_fr - self.now_fr < self.done_fr
            ):
                self.status = "finished"

        return None

    def _step_impl(self) -> bool:
        raise NotImplementedError("Subclasses must implement this method")
