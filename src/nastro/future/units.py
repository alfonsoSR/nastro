from astropy import units as u
from typing import Any
from .types import Double, Vector, is_double, is_vector


class Quantity[U: (Double, Vector)](u.Quantity):

    def __init__(self, value: U, unit: u.Unit | str) -> None:

        super().__init__(value, unit=unit)  # type: ignore

        return None

    @property
    def value(self) -> U:
        return super().value  # type: ignore


_m: Any = u.m  # type: ignore
m: u.Unit = _m
