from typing import Any
from .. import types as nt


class Force:
    """Base class for single forces"""

    def __init__(self, *args, **kwargs) -> None:
        raise NotImplementedError

    def __call__(
        self, t: nt.Double, s: nt.CartesianState, fr: nt.Double = 0.0
    ) -> nt.CartesianStateDerivative:
        raise NotImplementedError


class ForceModel:
    """Complete force model to be passed to a the solver"""

    def __init__(self, *forces: Force) -> None:

        self.forces = forces
        self.n_forces = len(self.forces)
        return None

    def __call__(
        self, t: nt.Double, s: nt.CartesianState, fr: nt.Double = 0.0
    ) -> nt.CartesianStateDerivative:

        ds = self.forces[0](t, s, fr)
        if self.n_forces > 1:
            for i in range(1, self.n_forces):
                ds.add_acceleration(self.forces[i](t, s, fr))

        return ds
