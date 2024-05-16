from . import core as nt
from datetime import datetime
import numpy as np
from typing import Any, Iterator, Sequence, Self, Union, Literal
from pathlib import Path
from ..constants import day
import traceback

# from ..data.formats import EOP


class JulianDay[T: (nt.Double, nt.Vector)]:
    """Julian day

    .. warning:: You might pass the integral and fractional parts of the Julian day as a single number, but this will result in a loss of precision due to floating point arithmetic.

    A Julian Day is specified by an integral part (day), which corresponds to
    12:00:00 of a certain day, and a fractional part (time), which represents the
    amount of time that has passed since that moment as a fraction of a day.

    NOTE: Accurate to 16 decimal places if integral part ends in 0.5 and to 15 - int(log_10(jd_int)) decimal places otherwise. For example, with (2451545.5, None) as input, the fractional part is accurate to 16 decimal places, but with (2451545.76, None) as input, the fractional part is accurate to 10 decimal places.
    """

    def __init__(
        self,
        jd_int: T,
        jd_frac: T | None = None,
        ref: Literal["J2000", "MJD"] | None = None,
    ) -> None:

        # Get type
        if nt.is_double(jd_int):
            if jd_frac is not None and not nt.is_double(jd_frac):
                raise ValueError("Failed to initialize JulianDay: Invalid input")
            self.scalar = True
        elif isinstance(jd_int, np.ndarray):
            if isinstance(jd_frac, np.ndarray) and jd_frac.dtype == np.dtype("O"):
                raise ValueError("Failed to initialize JulianDay: Invalid input")
            self.scalar = False
        else:
            raise ValueError("Failed to initialize JulianDay: Invalid input type")

        # Generate array containers
        q1: Any = np.array([jd_int], dtype=np.float64).ravel()
        if jd_frac is None:
            q2: Any = np.zeros_like(q1)
        else:
            q2: Any = np.array([jd_frac], dtype=np.float64).ravel()

        if q1.size != q2.size:
            raise ValueError("Integral and fractional parts must have the same size")

        # Add reference epoch if necessary
        if ref is not None:

            if ref == "J2000":
                q1 += 2451545.0
            elif ref == "MJD":
                q1 += 2400000.0
                q2 += 0.5
            else:
                raise ValueError(
                    "Failed to initialize JulianDay: Invalid reference epoch"
                )

        self.q1, self.q2 = self.__split(q1, q2)

        return None

    def __split(self, jd_int: Any, jd_frac: Any) -> tuple[Any, Any]:
        """Split Julian date into integral and fractional parts"""

        day = np.round(jd_int) - 0.5
        time = jd_int - day
        dI, dF = np.divmod(jd_frac, 1.0)
        ddI, dF = np.divmod(time + dF, 1.0)
        return day + dI + ddI, dF

    @property
    def size(self) -> int:
        return self.q1.size

    @property
    def day(self) -> T:
        """Integral part of the Julian Day"""
        return self.q1[0] if self.scalar else self.q1

    @property
    def time(self) -> T:
        """Fractional part of the Julian Day"""
        return self.q2[0] if self.scalar else self.q2

    @property
    def jd(self) -> T:
        """Julian date and time"""
        return self.day + self.time

    @property
    def mday(self) -> T:
        """Integral part of modified Julian Day"""
        return (self - 2400000.5).day

    @property
    def mtime(self) -> T:
        """Fractional part of modified Julian Day"""
        return (self - 2400000.5).time

    @property
    def mjd(self) -> T:
        """Modified Julian date and time"""
        return (self - 2400000.5).jd

    @property
    def dt(self) -> T:
        """Days past initial epoch"""
        return self.jd - self[0].jd

    def __repr__(self) -> str:

        out = "Julian Day\n"
        out += "-" * len(out) + "\n"
        out += f"Day: {self.day}\n"
        out += f"Time: {self.time}\n"
        return out

    def __sub__(self, other: object) -> Union["JulianDay[T]", "DeltaJD[T]"]:

        if nt.is_double(other):
            dI, dF = np.divmod(other, 1.0)
            return JulianDay(self.day - dI, self.time - dF)
        elif isinstance(other, np.ndarray) and other.dtype != np.dtype("O"):
            if other.size != self.size:
                raise ValueError("Failed to perform subtraction: Sizes do not match")
            dI, dF = np.divmod(other, 1.0)
            day: Any = self.day - dI
            time: Any = self.time - dF
            return JulianDay(day, time)
        elif isinstance(other, self.__class__):
            if self.size != other.size:
                raise ValueError("Failed to perform subtraction: Sizes do not match")

            delta_day = self.day - other.day
            delta_time = self.time - other.time
            return DeltaJD(delta_day, delta_time)
        else:
            raise TypeError("Failed to perform subtraction. Invalid type")

    def __add__(self, other: object) -> "JulianDay[T]":

        if nt.is_double(other):
            dI, dF = np.divmod(other, 1.0)
            return JulianDay(self.day + dI, self.time + dF)
        elif isinstance(other, np.ndarray) and other.dtype != np.dtype("O"):
            if other.size != self.size:
                raise ValueError("Failed to perform addition: Sizes do not match")
            dI, dF = np.divmod(other, 1.0)
            day: Any = self.day + dI
            time: Any = self.time + dF
            return JulianDay(day, time)
        else:
            raise TypeError("Failed to perform addition. Invalid type")

    def __getitem__(self, index: int | slice) -> "JulianDay[T]":
        return JulianDay(self.q1[index], self.q2[index])

    def __iter__(self) -> Iterator["JulianDay[T]"]:
        return iter([self.__getitem__(idx) for idx in range(self.size)])

    def __eq__(self, other: object) -> bool:
        assert isinstance(other, JulianDay)
        same_day = np.allclose(self.day, other.day, atol=1e-15, rtol=0.0)
        same_time = np.allclose(self.time, other.time, atol=1e-15, rtol=0.0)
        return same_day and same_time

    def split(self, gap: nt.Double) -> tuple[Sequence[int], Sequence[Self]]:
        """Split time series of Julian dates into intervals

        :param gap: Maximum gap between consecutive epochs in the same interval
        :return: Tuple containing indices of split points and sequence of intervals
        """

        delta = self[1:] - self[:-1]
        idx_list = np.argwhere(delta.jd > gap).ravel() + 1
        passes = []
        low = 0
        for idx in idx_list:
            passes.append(self[low:idx])
            low = idx
        passes.append(self[low:])
        idx_list = [0] + idx_list.tolist()
        return idx_list, passes

    def as_calendar(self) -> "CalendarDate" | Sequence["CalendarDate"]:
        """Conversion to calendar date and time

        Algorithm: Practical astrodynamics slides - 2.17
        """

        jd0 = self.day + 0.5
        L1 = np.trunc(jd0 + 68569)
        L2 = np.trunc(4 * L1 / 146097)
        L3 = L1 - np.trunc((146097 * L2 + 3) / 4)
        L4 = np.trunc(4000 * (L3 + 1) / 1461001)
        L5 = L3 - np.trunc(1461 * L4 / 4) + 31
        L6 = np.trunc(80 * L5 / 2447)
        L7 = np.trunc(L6 / 11)
        day = L5 - np.trunc(2447 * L6 / 80)
        month = L6 + 2 - 12 * L7
        year = 100 * (L2 - 49) + L4 + L7
        hour = np.trunc(self.time * 24)
        rem = self.time * 24 - hour
        minute = np.trunc(rem * 60)
        rem = rem * 60 - minute
        second = np.trunc(rem * 60)
        micro_second = np.trunc((rem * 60 - second) * 1e6)

        if nt.is_double(self.day):
            return CalendarDate(
                int(year),
                int(month),
                int(day),
                int(hour),
                int(minute),
                int(second),
                int(micro_second),
            )

        elif isinstance(self.day, np.ndarray):
            return [
                CalendarDate(
                    int(year[i]),
                    int(month[i]),
                    int(day[i]),
                    int(hour[i]),
                    int(minute[i]),
                    int(second[i]),
                    int(micro_second[i]),
                )
                for i in range(self.day.size)
            ]
        else:
            raise TypeError("Failed to convert to calendar date: Invalid type")

    def save(self, path: str | Path) -> None:
        """Save Julian date to Numpy binary file

        :param path: Path to file to save
        """
        if isinstance(path, str):
            path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.array([self.day, self.time]))
        return None

    @classmethod
    def load(
        cls,
        path: str | Path,
        ref: Literal["J2000", "MJD"] | None = None,
        unit: Literal["s", "d"] = "d",
        relative: bool = True,
    ) -> Self:
        """Load Julian date from Numpy binary file

        :param path: Path to file to load
        """

        # If extension is missing, assume .npy
        path = Path(path)
        if not path.suffix:
            path = path.with_suffix(".npy")

        # Generate absolute path
        cwd = Path("/")
        if relative:
            cwd = Path(traceback.extract_stack()[-2].filename).parent
        path = cwd / path

        # Load output
        match path.suffix:
            case ".npy":
                return cls(*np.load(path), ref=ref)
            case ".dat":
                data = np.loadtxt(path).T
                if unit == "s":
                    data /= day
                if data.shape[0] == 1 or data.shape[0] == 7:
                    return cls(data[0], ref=ref)
                elif data.shape[0] == 2 or data.shape[0] == 8:
                    return cls(data[0], data[1], ref=ref)
                else:
                    raise ValueError("Failed to load Julian date: Invalid data format")
            case _:
                raise ValueError("Failed to load Julian date: Invalid file extension")

    @classmethod
    def from_tudat(
        cls,
        state_history: dict[nt.Double, list[nt.Double]],
        ref: Literal["J2000", "MJD"] | None = None,
        unit: Literal["s", "d"] = "d",
    ) -> Self:
        """Load Julian date from TUDAT state history

        :param state_history: Dictionary containing state history data
        """
        data = np.array(list(state_history.keys())) / day
        return cls(data, ref=ref)


class DeltaJD[T: (nt.Double, nt.Vector)]:
    """Difference between two Julian days

    NOTE: You are not supposed to instantiate DeltaJD. You can subtract Double or
    Vector types directly from a JulianDay object.

    :param dI: Integral part of the difference
    :param dF: Fractional part of the difference
    """

    def __init__(self, dI: T, dF: T) -> None:

        # Get typea
        if nt.is_double(dI):
            if dF is not None and not nt.is_double(dF):
                raise ValueError("Failed to initialize JulianDay: Invalid input")
            self.scalar = True
        elif isinstance(dI, np.ndarray):
            if not isinstance(dF, np.ndarray) or dF.dtype == np.dtype("O"):
                raise ValueError("Failed to initialize JulianDay: Invalid input")
            self.scalar = False
        else:
            raise ValueError("Failed to initialize JulianDay: Invalid input type")

        q1: Any = np.array([dI], dtype=np.float64).ravel()
        q2: Any = np.array([dF], dtype=np.float64).ravel()

        if q1.size != q2.size:
            raise ValueError("Integral and fractional parts must have the same size")

        self.q1, self.q2 = self.__split(q1, q2)

        return None

    def __split(self, dI: Any, dF: Any) -> tuple[Any, Any]:
        """Split Julian date into integral and fractional parts"""

        ddI = np.round(dI)
        ddF = dI - ddI
        deltaI, deltaF = np.divmod(dF, np.where(dF < 0, -1.0, 1.0))
        ddF = ddF + deltaF
        ddeltaI, deltaF = np.divmod(ddF, np.where(ddF < 0, -1.0, 1.0))
        return ddI + deltaI + ddeltaI, deltaF

    @property
    def size(self) -> int:
        return self.q1.size

    @property
    def day(self) -> T:
        return self.q1[0] if self.scalar else self.q1

    @property
    def time(self) -> T:
        return self.q2[0] if self.scalar else self.q2

    @property
    def jd(self) -> T:
        return self.day + self.time


class CalendarDate(datetime):
    """Base class for calendar date types"""

    def as_list(self) -> list[int]:
        return [self.year, self.month, self.day, self.hour, self.minute, self.second]

    def as_jd(self) -> "JulianDay":
        """Conversion to Julian date and time

        Algorithm: Practical astrodynamics slides - 2.16
        """

        C = np.trunc((self.month - 14) / 12)
        jd0 = self.day - 32075 + np.trunc(1461 * (self.year + 4800 + C) / 4)
        jd0 += np.trunc(367 * (self.month - 2 - C * 12) / 12)
        jd0 -= np.trunc(3 * np.trunc((self.year + 4900 + C) / 100) / 4)
        jd = jd0 - 0.5
        fr = self.hour / 24.0 + self.minute / 1440.0 + self.second / 86400.0

        return JulianDay(jd, fr)


# class JulianDate[T: (Double, Vector)]:
#     """Julian date and time"""

#     def __init__(self, jd: T, fr: T | None = None) -> None:

#         self.scalar = False
#         if isinstance(jd, Double):
#             self.scalar = True

#         self.q1: Any = np.array([jd], dtype=np.float64).ravel()
#         if fr is None:
#             self.q2: Any = np.zeros_like(self.q1)
#         else:
#             self.q2: Any = np.array([fr], dtype=np.float64).ravel()

#         return None

#     @property
#     def size(self) -> int:
#         return self.q1.size

#     @property
#     def day(self) -> T:
#         return self.q1[0] if self.scalar else self.q1

#     @property
#     def frac(self) -> T:
#         return self.q2[0] if self.scalar else self.q2

#     @property
#     def mint(self) -> T:
#         return self.day - 2400000.5

#     @property
#     def jd(self) -> T:
#         return self.day + self.frac

#     @property
#     def mjd(self) -> T:
#         return self.mint + self.frac

#     def __eq__(self, other: object) -> bool:
#         assert isinstance(other, JulianDate)
#         is_this_jd = (self.day == other.day) and (self.frac == other.frac)
#         return is_jd and is_this_jd

#     def __repr__(self) -> str:

#         limit = 5 if self.size > 5 else self.size
#         out = "Integral:\t"
#         for i in range(limit):
#             out += f"{self.q1[i]:.15e} "
#         if self.size > 5:
#             out += "..."
#         out += "\nFractional:\t"
#         for i in range(limit):
#             out += f"{self.q2[i]:.15e} " if self.q2[i] != 0.0 else "0 "
#         if self.size > 5:
#             out += "..."
#         return out

#     def __add__(self, other: Self) -> "JulianDate[T]":

#         return JulianDate(self.day + other.day, self.frac + other.frac)

#     def __sub__(self, other: Self) -> "JulianDate":

#         return JulianDate(self.day - other.day, self.frac - other.frac)

#     def __getitem__(self, index: int | slice) -> "JulianDate":

#         return JulianDate(self.q1[index], self.q2[index])

#     def __iter__(self) -> Iterator["JulianDate"]:

#         return iter([self.__getitem__(idx) for idx in range(self.size)])

#     def as_date(self) -> Date | Sequence[Date]:
#         """Conversion to calendar date and time

#         Algorithm: Practical astrodynamics slides - 2.17
#         """

#         jd0 = self.day + 0.5
#         L1 = np.trunc(jd0 + 68569)
#         L2 = np.trunc(4 * L1 / 146097)
#         L3 = L1 - np.trunc((146097 * L2 + 3) / 4)
#         L4 = np.trunc(4000 * (L3 + 1) / 1461001)
#         L5 = L3 - np.trunc(1461 * L4 / 4) + 31
#         L6 = np.trunc(80 * L5 / 2447)
#         L7 = np.trunc(L6 / 11)
#         day = L5 - np.trunc(2447 * L6 / 80)
#         month = L6 + 2 - 12 * L7
#         year = 100 * (L2 - 49) + L4 + L7
#         hour = np.trunc(self.frac * 24)
#         rem = self.frac * 24 - hour
#         minute = np.trunc(rem * 60)
#         rem = rem * 60 - minute
#         second = np.round(rem * 60)
#         micro_second = np.trunc((rem * 60 - second) * 1e6)

#         if isinstance(self.day, Double):
#             return Date(
#                 int(year),
#                 int(month),
#                 int(day),
#                 int(hour),
#                 int(minute),
#                 int(second),
#                 int(micro_second),
#             )

#         elif isinstance(self.day, np.ndarray):
#             return [
#                 Date(
#                     int(year[i]),
#                     int(month[i]),
#                     int(day[i]),
#                     int(hour[i]),
#                     int(minute[i]),
#                     int(second[i]),
#                     int(micro_second[i]),
#                 )
#                 for i in range(self.day.size)
#             ]
#         else:
#             raise TypeError("Unexpected type for Julian date")

#     def split(self, gap: Double) -> tuple[list[int], list["JulianDate"]]:
#         """Split time series of Julian dates into intervals

#         :param gap: Maximum gap between consecutive epochs
#         """
#         idx_list = np.argwhere(np.diff(self.jd) > gap).ravel() + 1
#         passes = []
#         low = 0
#         for idx in idx_list:
#             passes.append(self[low:idx])
#             low = idx
#         return idx_list.tolist(), passes

#     @classmethod
#     def load(cls, path: str | Path) -> Self:
#         """Load Julian date from Numpy binary file

#         :param path: Path to file to load
#         """
#         data = np.load(path)
#         return cls(data)


class UTC(CalendarDate):
    pass

    # def as_ut1(self) -> "UT1":

    #     offset = EOP.at_epoch(self).ut1_utc
    #     print(offset)
    #     exit(0)


class UT1(CalendarDate):
    pass
