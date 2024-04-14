from ..types import CalendarDate as Date
from ..data import EOP


class ICRF:
    pass


class J2000:
    pass


class GECS:
    pass


class GCRF:
    pass


class ECI(GCRF):
    pass


class ECEF:

    def to_eci(self, utc_epoch: Date):

        eop = EOP.at_epoch(utc_epoch)

        return None


class ITRF(ECEF):
    pass


class TopocentricSEZ:
    pass


class RSW:
    pass
