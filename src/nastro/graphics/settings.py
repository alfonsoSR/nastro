from dataclasses import dataclass
from typing import Literal, Optional
from pathlib import Path


@dataclass
class PlotSetup:

    # Canvas configuration
    canvas_color: str | None = None
    canvas_size: tuple[float, float] = (7, 4)
    canvas_layout: Literal["tight", "constrained", "none", "compressed"] = (
        "constrained"
    )
    canvas_title: str | None = None

    # Figure configuration
    subfigure_color: Optional[str] = None
    subfigure_edgecolor: Optional[str] = None
    subfigure_edgewidth: float = 0.0
    subfigure_title: Optional[str] = None
    h_padding: float = 25 / 72
    w_padding: float = 30 / 72

    # Basic figure configuration
    figsize: tuple[float, float] = (7, 4)
    layout: Literal["tight", "constrained", "none", "compressed"] = "compressed"
    aspect: Literal["auto", "equal"] = "auto"
    figcolor: str | None = None
    title: str | None = None

    # Save and show
    show: bool = True
    save: bool = False
    dir: Path | str | None = None
    name: str | None = None

    # Subplot configuration
    axtitle: str | None = None

    xlabel: str | None = None
    ylabel: str | None = None
    zlabel: str | None = None
    rlabel: str | None = None
    plabel: str | None = None

    xscale: Literal["linear", "log", "symlog", "logit"] = "linear"
    yscale: Literal["linear", "log", "symlog", "logit"] = "linear"
    zscale: Literal["linear", "log", "symlog", "logit"] = "linear"
    rscale: Literal["linear", "log", "symlog", "logit"] = "linear"
    pscale: Literal["linear", "log", "symlog", "logit"] = "linear"

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    zlim: tuple[float, float] | None = None
    rlim: tuple[float, float] | None = None
    plim: tuple[float, float] | None = None

    scilimits: tuple[int, int] = (-2, 2)
    scilimits_x: Optional[tuple[int, int]] = None
    scilimits_y: Optional[tuple[int, int]] = None
    scilimits_z: Optional[tuple[int, int]] = None
    scilimits_r: Optional[tuple[int, int]] = None
    scilimits_p: Optional[tuple[int, int]] = None

    grid: bool = True
    grid_alpha: float = 0.15

    minor_ticks: bool = True
    minor_ticks_x: Optional[bool] = None
    minor_ticks_y: Optional[bool] = None
    minor_ticks_z: Optional[bool] = None
    show_tick_labels_x: bool = True
    show_tick_labels_y: bool = True
    show_tick_labels_z: bool = True
    show_tick_labels_r: bool = True
    show_tick_labels_p: bool = True
    custom_ticks: bool = False

    show_axes: bool = True

    legend: bool = True
    legend_location: str = "best"
    legend_title: str | None = None
    legend_columns: int = 1
    colorbar: bool = True
    colorbar_title: str | None = None
    colorbar_shrink: float = 1.0

    projection: Literal["persp", "ortho"] = "ortho"

    def copy(self) -> "PlotSetup":
        return PlotSetup(**self.__dict__)

    def version(self, **params) -> "PlotSetup":

        # Get contents of original setup
        contents = self.__dict__.copy()

        # Update contents with input
        for param, value in params.items():

            if param in contents:
                contents[param] = value
            else:
                ValueError(f"Invalid parameter: {param}")

        return PlotSetup(**contents)

        # new = self.copy()
        # for param, value in params.items():
        #     if hasattr(new, param):
        #         setattr(new, param, value)
        #     else:
        #         raise ValueError(f"Invalid parameter: {param}")

        # return new

    def __post_init__(self) -> None:

        # Minor ticks
        if self.minor_ticks_x is None:
            self.minor_ticks_x = self.minor_ticks
        if self.minor_ticks_y is None:
            self.minor_ticks_y = self.minor_ticks
        if self.minor_ticks_z is None:
            self.minor_ticks_z = self.minor_ticks

        # Scientific notation in labels
        if self.scilimits_x is None:
            self.scilimits_x = self.scilimits
        if self.scilimits_y is None:
            self.scilimits_y = self.scilimits
        if self.scilimits_z is None:
            self.scilimits_z = self.scilimits
        if self.scilimits_r is None:
            self.scilimits_r = self.scilimits
        if self.scilimits_p is None:
            self.scilimits_p = self.scilimits

        return None
