from astropy import time, coordinates
from ..types import Double, Vector, is_double, is_vector, is_array
from typing import Any
import datetime
import numpy as np


class Time[U: (Double, Vector)](time.Time):

    def __new__(
        cls,
        val,
        val2=None,
        format=None,
        scale=None,
        precision=None,
        in_subfmt=None,
        out_subfmt=None,
        location=None,
        copy=False,
    ) -> "Time[U]":

        self: "Time[U]" = super().__new__(
            cls,
            val,
            val2,
            format,
            scale,
            precision,
            in_subfmt,
            out_subfmt,
            location,
            copy,
        )  # type: ignore
        return self

    # Redefine properties with proper type hints
    @property
    def tai(self) -> "Time[U]":
        return Time(super().tai)  # type: ignore

    @property
    def tcb(self) -> "Time[U]":
        return Time(super().tcb)  # type: ignore

    @property
    def tcg(self) -> "Time[U]":
        return Time(super().tcg)  # type: ignore

    @property
    def tdb(self) -> "Time[U]":
        return Time(super().tdb)  # type: ignore

    @property
    def tt(self) -> "Time[U]":
        return Time(super().tt)  # type: ignore

    @property
    def ut1(self) -> "Time[U]":
        return Time(super().ut1)  # type: ignore

    @property
    def utc(self) -> "Time[U]":
        return Time(super().utc)  # type: ignore

    @property
    def datetime(self) -> datetime.datetime:
        return super().datetime  # type: ignore

    @property
    def delta_ut1_utc(self) -> U:
        val: Any = super().delta_ut1_utc
        return float(val) if self.isscalar else val  # type: ignore

    @property
    def delta_tdb_tt(self) -> U:
        val: Any = super().delta_tdb_tt
        return float(val) if self.isscalar else val  # type: ignore

    @property
    def gps(self) -> U:
        return super().gps  # type: ignore

    @property
    def jd(self) -> U:
        return super().jd  # type: ignore

    @property
    def jd1(self) -> U:
        return super().jd1  # type: ignore

    @property
    def jd2(self) -> U:
        return super().jd2  # type: ignore

    @property
    def jyear(self) -> U:
        return super().jyear  # type: ignore

    @property
    def mjd(self) -> U:
        return super().mjd  # type: ignore

    @property
    def unix(self) -> U:
        return super().unix  # type: ignore

    @property
    def unix_tai(self) -> U:
        return super().unix_tai  # type: ignore

    @staticmethod
    def now() -> "Time[U]":
        return Time(time.Time.now())

    def strftime(self, format_spec: str) -> np.ndarray:
        return super().strftime(format_spec)

    @classmethod
    def strptime(cls, time_string: str, format_string: str) -> "Time[U]":
        return Time(super().strptime(time_string, format_string))
