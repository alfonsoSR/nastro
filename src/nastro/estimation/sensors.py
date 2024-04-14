from ..types import Vector, Double, CartesianPosition


class Limits:

    min: Double
    max: Double


class Radar:

    def __init__(
        self,
        coordinates: CartesianPosition,
        range_range: tuple[Double, Double],
        elevation_range: tuple[Double, Double],
        azimuth_range: tuple[Double, Double],
        horizontal_FOV: tuple[Double, Double],
        vertical_FOV: tuple[Double, Double],
    ) -> None:

        self.coordinates = coordinates
        self.rho_lims = Limits(*range_range)
        self.el_lims = Limits(*elevation_range)
        self.az_lims = Limits(*azimuth_range)
        self.hfov_lims = Limits(*horizontal_FOV)
        self.vfov_lims = Limits(*vertical_FOV)

        return None
