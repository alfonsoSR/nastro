from .base import CartesianPropagator


class EulerPropagator(CartesianPropagator):

    def __init__(self, fun, t0, s0, tend, dt) -> None:

        super().__init__(fun, t0, s0, tend, dt)

        return None

    def _step_impl(self):

        new_time_int, new_time_fr = self._update_epoch()
        h_int, h_fr = self._get_steps()

        f = self.model(self.now_int, self.s, fr=self.now_fr)
        new_s = self.s + f * h_int + f * h_fr

        # new_s = self.s + self.fun(self.now_int, self.s, fr=self.now_fr) * h

        self.now_int = new_time_int
        self.now_fr = new_time_fr
        self.s = new_s

        return True
