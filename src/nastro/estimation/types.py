from ..types import Vector, Double


class RadarObservation[T: (Double, Vector)]:

    def __init__(
        self, epoch: T, range: Double, azimuth: Double, elevation: Double
    ) -> None:
        self.epoch = epoch
        self.range = range
        self.azimuth = azimuth
        self.elevation = elevation

        return None
