from ..types.core import Vector
from ..types.time import UTC
from ..constants import arcsec
import numpy as np
from pathlib import Path


class EOP:
    """Earth Orientation Parameters

    Attributes
    ----------
    xp, yp : float
        Celestial pole coordinates
    ut1_utc : float
        Difference between UT1 and UTC [includes leap seconds]
    lod : float
        Length of day
    ddpsi, ddeps : float
        Nutation corrections
    dx, dy : float
        Celestial pole offsets

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
    def at_epoch(cls, utc: UTC) -> "EOP":
        """Earth Orientation Parameters at epoch

        Retrieve EOP data at given epoch by computing a linear interpolation of
        parameters from the two closest times in the EOP data file.

        :param utc: UTC date and time
        """

        # Approximate the MJD of the input date
        epoch = utc.as_jd()
        mjd_int = int(epoch.mjd)

        # Find closest epochs in EOP data
        eop_data = np.load(Path(__file__).parent / "eop.npy").T
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

        return cls(interp_eop, tai_utc)
