from ..types.time import JulianDay, CalendarDate
from ..types.core import Vector
import numpy as np
from pathlib import Path
from ..constants import arcsec
from typing import TypeVar
from datetime import datetime


THIS_FILE = Path(__file__)


class EOP:
    """Earth Orientation Parameters

    Sources
    -------
    https://ggos.org/item/earth-orientation-parameter
    """

    def __init__(self, eop_data: Vector, tai_utc: int | None = None) -> None:

        if tai_utc is None:
            tai_utc = int(eop_data[-1])

        self.xp = eop_data[0] * arcsec
        self.yp = eop_data[1] * arcsec
        self.ut1_utc = eop_data[2] + tai_utc
        self.lod = eop_data[3]
        self.ddpsi = eop_data[4] * arcsec
        self.ddeps = eop_data[5] * arcsec
        self.dx = eop_data[6] * arcsec
        self.dy = eop_data[7] * arcsec

        return None

    @classmethod
    def at_epoch(cls, utc: CalendarDate) -> "EOP":
        """Earth Orientation Parameters at epoch

        Retrieve EOP data at given epoch by computing a linear interpolation of
        parameters from the two closest times in the EOP data file.

        :param utc: UTC date and time
        """

        # Approximate the MJD of the input date
        epoch = utc.as_jd()
        mjd_int = int(epoch.mjd)

        # Find closest epochs in EOP data
        eop_data = np.load(THIS_FILE.parent / "eop.npy").T
        idx_lower = np.nonzero(eop_data[3] >= mjd_int)[0][0]
        idx_upper = idx_lower + 1
        lower = eop_data[:, idx_lower]
        upper = eop_data[:, idx_upper]

        # Adjust UT1-UTC column in case leap second occurs between lines
        tai_utc = int(lower[-1])
        lower[3] -= lower[9]
        upper[3] -= upper[9]

        # Linear interpolation
        dt = epoch.mjd - lower[0]
        delta_eop = upper - lower
        interp_eop = lower[1:] + dt * delta_eop[1:] / delta_eop[0]

        # Convert output to arcseconds
        return EOP(interp_eop, tai_utc)


class XYS:

    def __init__(self, xys_data: Vector) -> None:

        self.year = xys_data[0]
        self.month = xys_data[1]
        self.day = xys_data[2]
        self.mjd = xys_data[3]
        self.x = xys_data[4]
        self.y = xys_data[5]
        self.s = xys_data[6]

        return None

    @classmethod
    def interpolate(cls, x: Vector, y: Vector, xx: float, order: int = 11):

        points = order + 1

        if len(x) < points:
            raise ValueError("Not enough data points for interpolation")

        # Compute number of elements on either side of middle element to grab
        nn, rem = np.divmod(points, 2)

        # Find index such that x[row0] < xx < x[row0 + 1]
        row0 = np.nonzero(x < xx)[0][-1]

        # Trim data set
        if rem == 0:
            # Adjust row0 in case near data set endpoints
            if (points == len(x)) or (row0 < nn - 1):
                row0 = nn - 1
            elif row0 > (len(x) - nn):
                row0 = len(x) - nn - 1

            # Trim to relevant data points
            x_trimed = x[row0 - nn + 1 : row0 + nn + 1]
            y_trimed = y[:, row0 - nn + 1 : row0 + nn + 1]
        else:
            # Adjust row0 in case near data set endpoints
            if (points == len(x)) or (row0 < nn):
                row0 = nn
            elif row0 > len(x) - nn:
                row0 = len(x) - nn - 1
            else:
                if (xx - x[row0] > 0.5) and (row0 + 1 + nn < len(x)):
                    row0 += 1

            # Trim to relevant data points
            x_trimed = x[row0 - nn : row0 + nn + 1]
            y_trimed = y[:, row0 - nn : row0 + nn + 1]

        # Compute coefficients (Didn't vectorize due to floating point)
        Pj = np.ones((1, points))

        for jj in range(points):
            for ii in range(points):
                if jj != ii:
                    Pj[0, jj] = (
                        Pj[0, jj] * (x_trimed[ii] - xx) / (x_trimed[ii] - x_trimed[jj])
                    )

        return np.dot(Pj, y_trimed.T)[0]

    @classmethod
    def at_epoch(cls, tt: JulianDay):

        xys_data = np.load(THIS_FILE.parent / "sources/xys.npy")

        # Compute MJD and round to nearest integer
        mjd_int = int(tt.mjd)

        # Number of additional data points to include on either side
        n = 10

        # Find closest epochs in XYS data
        mjd_data = xys_data[3]
        if mjd_int not in mjd_data:
            raise ValueError("Requested epoch not in XYS time range")

        mjd_idx = np.nonzero(mjd_data == mjd_int)[0][0]
        low = 0 if mjd_int <= mjd_data[0] + n else mjd_idx - n
        high = -1 if mjd_int > mjd_data[0] - n else mjd_idx + n
        xys_data = xys_data[:, low:] if high == -1 else xys_data[:, low:high]

        x_data = xys_data[3]
        y_data = xys_data[4:]
        x, y, s = cls.interpolate(x_data, y_data, tt.mjd, 11) * arcsec
        return x, y, s
