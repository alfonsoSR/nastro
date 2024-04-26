from ..forces import ForceModel
from ..types import CartesianState, JulianDay, Double
import numpy as np
from ..constants import day
from typing import Any


class CartesianPropagator[U: (Double, JulianDay)]:
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
        self, model: ForceModel, t0: U, s0: CartesianState[Double], tend: U, dt: Double
    ) -> None:

        self.model = model

        # Initial and final epoch and step
        if isinstance(t0, JulianDay) and isinstance(tend, JulianDay):
            self.is_day = True
            self.now_int, self.now_fr = t0.day, t0.time
            self.tend_int, self.tend_fr = tend.day, tend.time
            dt /= day
        elif isinstance(t0, Double) and isinstance(tend, Double):
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
            self.h_sec = self.h_int * day + self.h_fr * day
        else:
            self.h_sec = self.h_int + self.h_fr

        # Termination criteria
        self.EPS = 1e-14
        self.done_int = max(self.EPS, 0.1 * self.h_int)
        self.done_fr = max(self.EPS, 0.1 * self.h_fr)

        # Status
        self.status = "running"

        return None

    def _update_epoch(self) -> tuple[Double, Double]:

        new_fr = self.now_fr + self.h_fr
        int_part = np.trunc(new_fr)
        new_fr -= int_part
        new_int = self.now_int + self.h_int + int_part

        return new_int, new_fr

    def _get_steps(self) -> tuple[Double, Double]:

        if self.is_day:
            return self.h_int * day, self.h_fr * day
        else:
            return self.h_int, self.h_fr

    def propagate(self, frac: bool = True) -> tuple[U, CartesianState]:
        """Propagate initial state"""

        out_sol = CartesianState(*self.s.asarray())
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
        time: Any = JulianDay(time_int, time_fr) if self.is_day else time_int + time_fr

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
