from typing import Literal
from .base import CartesianPropagator
from ..types import CartesianState, CartesianStateDerivative, Double


class RungeKuttaPropagator(CartesianPropagator):

    def __init__(self, fun, t0, s0, tend, dt) -> None:

        super().__init__(fun, t0, s0, tend, dt)

        return None

    def _step_impl(self) -> bool:

        # TODO: Handle units properly (???? Apr 2024)

        new_time_int, new_time_fr = self._update_epoch()
        h_int, h_fr = self._get_steps()

        k1 = self.model(self.now_int, self.s, fr=self.now_fr)
        k2 = self.model(
            self.now_int + 0.5 * h_int,
            self.s + k1.times_dt(0.5 * h_int) + k1.times_dt(0.5 * h_fr),
            fr=self.now_fr + 0.5 * h_fr,
        )
        k3 = self.model(
            self.now_int + 0.5 * h_int,
            self.s + k2.times_dt(0.5 * h_int) + k2.times_dt(0.5 * h_fr),
            fr=self.now_fr + 0.5 * h_fr,
        )
        k4 = self.model(
            self.now_int + self.h_int,
            self.s + k3.times_dt(h_int) + k3.times_dt(h_fr),
            fr=self.now_fr + self.h_fr,
        )
        sum_k = (k1 + k2 * 2.0 + k3 * 2.0 + k4) * (1.0 / 6.0)

        new_s = self.s + sum_k.times_dt(h_int) + sum_k.times_dt(h_fr)

        self.now_int = new_time_int
        self.now_fr = new_time_fr
        self.s = new_s

        return True
