from matplotlib import pyplot as plt
from typing import (
    Literal,
    Self,
    Iterator,
    Any,
    TypeVar,
    TypeAlias,
    Sequence,
    Optional,
)
from matplotlib.gridspec import GridSpec, SubplotSpec
import numpy as np
from matplotlib.figure import SubFigure, Figure as mplFigure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D, axes3d
from matplotlib import _api

import matplotlib.cbook as cbook
from dataclasses import dataclass
from pathlib import Path
from matplotlib import ticker
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from ..types import Scalar, Array
from matplotlib.rcsetup import cycler
import matplotlib.transforms as mtra

PlotType = TypeVar("PlotType", bound="BaseFigure")
FigureLike: TypeAlias = SubFigure | mplFigure
AxesLike: TypeAlias = Axes | Axes3D
Plotable: TypeAlias = Scalar | Array | Sequence[Scalar]

COLOR_CYCLER: Any = cycler(  # type: ignore
    color=[
        "#1f77b4",
        "#aec7e8",
        "#ff7f0e",
        "#ffbb78",
        "#2ca02c",
        "#98df8a",
        "#d62728",
        "#ff9896",
        "#9467bd",
        "#c5b0d5",
        "#8c564b",
        "#c49c94",
        "#e377c2",
        "#f7b6d2",
        "#7f7f7f",
        "#c7c7c7",
        "#bcbd22",
        "#dbdb8d",
        "#17becf",
        "#9edae5",
    ]
)


@dataclass
class Artist:

    axis: str
    type: str
    color: Optional[str]
    object: Any


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


class Canvas:
    """Blank window in which stuff is drawn"""

    def __init__(self, mosaic: str, setup: PlotSetup) -> None:

        self.canvas_setup = setup
        self.canvas = plt.figure(
            figsize=self.canvas_setup.canvas_size,
            layout=self.canvas_setup.canvas_layout,
            facecolor=self.canvas_setup.canvas_color,
        )

        if self.canvas_setup.canvas_title is not None:
            self.canvas.suptitle(self.canvas_setup.canvas_title)

        gridspec, structure = self.__generate_mosaic(mosaic)
        self.canvas_gridspec = gridspec
        self.canvas_structure = structure

        # self.subfigures = iter(
        #     self.canvas.add_subfigure(gridspec[sti]) for sti in structure
        # )

        return None

    def __make_array(self, inp):
        """Array representation of mosaic string"""

        r0, *rest = inp
        if isinstance(r0, str):
            raise ValueError("List mosaic specification must be 2D")
        for j, r in enumerate(rest, start=1):
            if isinstance(r, str):
                raise ValueError("List mosaic specification must be 2D")
            if len(r0) != len(r):
                raise ValueError(
                    "All of the rows must be the same length, however "
                    f"the first row ({r0!r}) has length {len(r0)} "
                    f"and row {j} ({r!r}) has length {len(r)}."
                )
        out = np.zeros((len(inp), len(r0)), dtype=object)
        for j, r in enumerate(inp):
            for k, v in enumerate(r):
                out[j, k] = v

        return out

    def __identify_keys_and_nested(self, mosaic) -> Any:
        """FROM matplotlib.subplot_mosaic"""
        unique_ids = cbook._OrderedSet()
        nested = {}
        for j, row in enumerate(mosaic):
            for k, v in enumerate(row):
                if v == ".":
                    continue
                elif not cbook.is_scalar_or_string(v):
                    nested[(j, k)] = self.__make_array(v)
                else:
                    unique_ids.add(v)

        return tuple(unique_ids), nested

    def __do_layout(self, gs, mosaic, unique_ids, nested):
        """Generates figure layout from mosaic array"""

        this_level = dict()

        for name in unique_ids:
            indx = np.argwhere(mosaic == name)
            start_row, start_col = np.min(indx, axis=0)
            end_row, end_col = np.max(indx, axis=0) + 1
            slc = (slice(start_row, end_row), slice(start_col, end_col))
            if (mosaic[slc] != name).any():
                raise ValueError(
                    f"While trying to layout\n{mosaic!r}\n"
                    f"we found that the label {name!r} specifies a "
                    "non-rectangular or non-contiguous area."
                )
            this_level[(start_row, start_col)] = (name, slc, "axes")

        for (j, k), nested_mosaic in nested.items():
            this_level[(j, k)] = (None, nested_mosaic, "nested")

        return [this_level[key][1] for key in sorted(this_level)]

    def __generate_mosaic(self, mosaic) -> tuple[GridSpec, Iterator[slice]]:

        __mosaic = self.__make_array(
            self.canvas._normalize_grid_string(mosaic),  # type: ignore
        )
        rows, cols = __mosaic.shape
        gridspec = self.canvas.add_gridspec(rows, cols)
        layout = self.__do_layout(
            gridspec, __mosaic, *self.__identify_keys_and_nested(__mosaic)
        )
        return gridspec, iter(layout)

    def __enter__(self):

        # Don't make plots if they are not shown or saved
        if self.canvas_setup.show or self.canvas_setup.save:
            return self
        else:
            return NotImplemented

    def __exit__(self, exc_type, exc_value, traceback) -> bool:

        if exc_type is AttributeError and "NotImplementedType" in str(
            exc_value
        ):
            return True
        elif exc_type is not None:
            return False
        else:
            pass

        if self.canvas_setup.save:

            if self.canvas_setup.dir is None or self.canvas_setup.name is None:
                raise ValueError(
                    "Failed to save figure: missing filename or directory"
                )
            path = Path(self.canvas_setup.dir) / self.canvas_setup.name
            path.parent.mkdir(parents=True, exist_ok=True)
            self.canvas.savefig(path)

        if self.canvas_setup.show:
            plt.show()

        print(f"Canvas closed.")
        plt.close()

        return True


class BaseFigure(Canvas):

    _default_prefix: str = "ignored_artist"

    def __init__(
        self,
        setup: PlotSetup = PlotSetup(),
        _figure: FigureLike | None = None,
    ) -> None:

        self.setup = setup
        self.cycler = iter(COLOR_CYCLER)

        # Get figure and axes
        if _figure is None:
            self.is_subplot = False
            super().__init__("a", setup)
            self.figure = self.canvas
        else:
            self.is_subplot = True
            self.figure = _figure

        self.axes = {"left": self.generate_subplot()}

        # Setup figure and axes
        self.setup = setup
        self.custom_configuration()
        self.common_configuration()

        # Containers
        self.artists: dict[str, Artist] = {}

        return None

    def next_color(self) -> str:
        try:
            return next(self.cycler)["color"]
        except StopIteration:
            self.cycler = iter(COLOR_CYCLER)
            return next(self.cycler)["color"]

    def generate_subplot(self) -> AxesLike:
        return self.figure.add_subplot()

    def custom_configuration(self) -> None:
        return None

    def common_configuration(self) -> None:

        if self.setup.axtitle:
            self.axes["left"].set_title(self.setup.axtitle)

        # Labels
        if self.setup.xlabel:
            self.axes["left"].set_xlabel(self.setup.xlabel)
        if self.setup.ylabel:
            self.axes["left"].set_ylabel(self.setup.ylabel)
        if self.setup.zlabel and isinstance(self.axes["left"], Axes3D):
            self.axes["left"].set_zlabel(self.setup.zlabel)
        if self.setup.rlabel and "right" in self.axes:
            self.axes["right"].set_ylabel(self.setup.rlabel)
        if self.setup.plabel and "parasite" in self.axes:
            self.axes["parasite"].set_ylabel(self.setup.plabel)

        # Scales
        if self.setup.xscale:
            self.axes["left"].set_xscale(self.setup.xscale)
        if self.setup.yscale:
            if self.setup.yscale == "symlog":
                self.axes["left"].set_yscale(self.setup.yscale, linthresh=1e-12)
            else:
                self.axes["left"].set_yscale(self.setup.yscale)
        if self.setup.zscale and isinstance(self.axes["left"], Axes3D):
            self.axes["left"].set_zscale(self.setup.zscale)  # type: ignore
        if self.setup.rscale and "right" in self.axes:
            self.axes["right"].set_yscale(self.setup.rscale)
        if self.setup.pscale and "parasite" in self.axes:
            self.axes["parasite"].set_yscale(self.setup.pscale)

        # Limits
        if self.setup.xlim:
            self.axes["left"].set_xlim(self.setup.xlim)
        if self.setup.ylim:
            self.axes["left"].set_ylim(self.setup.ylim)
        if self.setup.zlim and isinstance(self.axes["left"], Axes3D):
            self.axes["left"].set_zlim(self.setup.zlim)
        if self.setup.rlim and "right" in self.axes:
            self.axes["right"].set_ylim(self.setup.rlim)
        if self.setup.plim and "parasite" in self.axes:
            self.axes["parasite"].set_ylim(self.setup.plim)

        # Grid and other configurations
        if self.setup.grid:
            self.axes["left"].grid(alpha=self.setup.grid_alpha, which="both")

        return None

    def common_postprocessing(self) -> None:

        # Colors
        self.cycler = iter(COLOR_CYCLER)
        for artist in self.artists.values():

            match artist.type:

                case "errorbar":
                    color = (
                        self.next_color()
                        if artist.color is None
                        else artist.color
                    )
                    artist.object[0].set_color(color)
                    for cap in artist.object[2]:
                        cap.set_color(color)

                case "step":
                    color = (
                        self.next_color()
                        if artist.color is None
                        else artist.color
                    )
                    for line in artist.object:
                        line.set_color(color)

                case "barh":
                    for bar in artist.object:
                        bar.set_color(self.next_color())

                case "bar":
                    for bar in artist.object:
                        bar.set_color(self.next_color())

                case "hist":
                    color = (
                        self.next_color()
                        if artist.color is None
                        else artist.color
                    )
                    for bar in artist.object:
                        bar.set_color(color)

                case "cmap":
                    if self.setup.colorbar:
                        self.figure.colorbar(
                            artist.object,
                            ax=self.axes["left"],
                            label=self.setup.colorbar_title,
                            shrink=self.setup.colorbar_shrink,
                        )

                case "image":
                    if self.setup.colorbar:
                        self.figure.colorbar(
                            artist.object,
                            ax=self.axes["left"],
                            label=self.setup.colorbar_title,
                            shrink=self.setup.colorbar_shrink,
                        )

                case "patch":
                    color = (
                        self.next_color()
                        if artist.color is None
                        else artist.color
                    )
                    artist.object.set_color(color)

                case "contour":
                    pass

                case "surface":
                    self.next_color()

                case _:
                    color = (
                        self.next_color()
                        if artist.color is None
                        else artist.color
                    )
                    artist.object.set_color(color)

        legend_handles = []
        for axis in self.axes.values():

            # Ticks and axes
            axis.tick_params(direction="in", which="both")
            if axis.get_yscale() == "linear":
                axis.ticklabel_format(
                    axis="y", scilimits=self.setup.scilimits_y, useMathText=True
                )
                if self.setup.minor_ticks_y:
                    axis.yaxis.set_minor_locator(ticker.AutoMinorLocator())
            if axis.get_xscale() == "linear":
                axis.ticklabel_format(
                    axis="x", scilimits=self.setup.scilimits_x, useMathText=True
                )
                if self.setup.minor_ticks_x:
                    axis.xaxis.set_minor_locator(ticker.AutoMinorLocator())

            # Do not show ticks if requested
            if not self.setup.show_tick_labels_x:
                axis.xaxis.set_major_formatter(ticker.NullFormatter())
                axis.xaxis.set_major_formatter(ticker.NullFormatter())
            if not self.setup.show_tick_labels_y:
                axis.yaxis.set_major_formatter(ticker.NullFormatter())
                axis.yaxis.set_minor_formatter(ticker.NullFormatter())

            if not self.setup.show_axes:
                axis.axis("off")

            # Legend
            for handle in axis.get_legend_handles_labels()[0]:
                if handle not in legend_handles:
                    legend_handles.append(handle)

            # Aspect ratio
            axis.set_aspect(self.setup.aspect)

        # Legend
        if self.setup.legend and legend_handles != []:
            __last_axis = list(self.axes.values())[-1]
            __last_axis.legend(
                loc=self.setup.legend_location,
                handles=legend_handles,
                title=self.setup.legend_title,
                ncols=self.setup.legend_columns,
            )

        return None

    def custom_postprocessing(self) -> None:
        return None

    def __enter__(self) -> Self:

        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool:

        self.common_postprocessing()
        self.custom_postprocessing()

        if not self.is_subplot:
            super().__exit__(exc_type, exc_value, traceback)

        if exc_type is not None:
            return False

        return True

    def __default_artist_label(self) -> str:

        return f"{self._default_prefix}{len(self.artists)}"

    def line(
        self,
        x: Array | Scalar,
        y: Optional[Array | Scalar] = None,
        z: Optional[Array | Scalar] = None,
        fmt: str = "-",
        width: Optional[float] = None,
        markersize: Optional[float] = None,
        color: Optional[str] = None,
        alpha: float = 1.0,
        label: Optional[str] = None,
        axis: str = "left",
    ) -> None:

        args = [x, y] if y is not None else [x]
        (line,) = self.axes[axis].plot(
            *args,
            fmt,
            linewidth=width,
            markersize=markersize,
            color=color,
            alpha=alpha,
            label=label,
        )

        name = label if label is not None else self.__default_artist_label()
        self.artists[name] = Artist(axis, "line", color, line)

        return None

    def limits(
        self,
        min: Scalar,
        max: Scalar,
        color: str | None = None,
        alpha: float = 0.1,
        label: str | None = None,
        axis: str = "left",
    ) -> None:

        min = float(min)
        max = float(max)

        boundary = self.axes[axis].add_artist(
            Rectangle(
                (-1e20, min),
                2e20,
                max - min,
                color=color,
                alpha=alpha,
                label=label,
            )
        )

        name = label if label is not None else self.__default_artist_label()
        self.artists[name] = Artist(axis, "limits", color, boundary)

        return None

    def boundary(
        self,
        error: Scalar | Array,
        reference: str | Literal["last"] = "last",
        color: str | None = None,
        alpha: float = 0.1,
        label: str | None = None,
        axis: str = "left",
    ) -> None:

        if reference == "last":
            alist = list(self.artists.values())
            for idx in range(1, len(self.artists) + 1):
                if alist[-idx].axis == axis and alist[-idx].type == "line":
                    reference = alist[-idx].object
                    break
        else:
            reference = self.artists[reference].object
        assert isinstance(reference, Line2D)

        x_data = reference.get_xdata()
        y_data = reference.get_ydata()

        boundary = self.axes[axis].fill_between(
            x_data,
            y_data - np.array(error),
            y_data + np.array(error),
            color=color,
            alpha=alpha,
            label=label,
        )  # type: ignore

        name = label if label is not None else self.__default_artist_label()
        self.artists[name] = Artist(axis, "limits", color, boundary)

        return None

    def vlimits(
        self,
        min: Scalar,
        max: Scalar,
        color: str | None = None,
        alpha: float = 0.1,
        label: str | None = None,
        axis: str = "left",
    ) -> None:

        min = float(min)
        max = float(max)

        boundary = self.axes[axis].add_artist(
            Rectangle(
                (min, -1e20),
                max - min,
                2e20,
                color=color,
                alpha=alpha,
                label=label,
            )
        )

        name = label if label is not None else self.__default_artist_label()
        self.artists[name] = Artist(axis, "vlimits", color, boundary)

        return None

    def errorbar(
        self,
        x: Array,
        y: Array,
        error: Scalar | Array,
        z: Optional[Array] = None,
        fmt="-",
        color: Optional[str] = None,
        label: Optional[str] = None,
        axis: str = "left",
    ) -> None:

        errorbar = self.axes[axis].errorbar(
            x, y, z=z, yerr=error, fmt=fmt, color=color, label=label
        )

        name = label if label is not None else self.__default_artist_label()
        self.artists[name] = Artist(axis, "errorbar", color, errorbar)

        return None

    def step(
        self,
        x: Array,
        y: Optional[Array] = None,
        where: Literal["pre", "post", "mid"] = "mid",
        fmt: str = "-",
        color: Optional[str] = None,
        label: Optional[str] = None,
        axis: str = "left",
    ) -> None:

        args = [x, y] if y is not None else [x]
        step = self.axes[axis].step(
            *args,
            fmt,
            where=where,
            color=color,
            label=label,
        )
        name = label if label is not None else self.__default_artist_label()
        self.artists[name] = Artist(axis, "step", color, step)

        return None

    def bar(
        self,
        x: Array,
        height: Array,
        width: float = 0.8,
        ticks: Optional[Array] = None,
        axis: str = "left",
    ) -> None:

        bar = self.axes[axis].bar(x, height, width=width, tick_label=ticks)
        name = self.__default_artist_label()
        self.artists[name] = Artist(axis, "bar", None, bar)

        return None

    def barh(
        self,
        y: Array,
        width: Array,
        height: float = 0.8,
        ticks: Optional[Array] = None,
        axis: str = "left",
    ) -> None:

        bar = self.axes[axis].barh(y, width, height=height, tick_label=ticks)
        name = self.__default_artist_label()
        self.artists[name] = Artist(axis, "barh", None, bar)

        return None

    def hist(
        self,
        data: Array,
        bins: int = 10,
        normalize: bool = False,
        cumulative: bool = False,
        hist_type: Literal["bar", "barstacked", "step", "stepfilled"] = "bar",
        align: Literal["left", "mid", "right"] = "mid",
        label: Optional[str] = None,
        color: Optional[str] = None,
        alpha: float = 0.8,
        axis: str = "left",
    ) -> None:

        _, _, histogram = self.axes[axis].hist(
            data,
            bins=bins,
            density=normalize,
            cumulative=cumulative,
            histtype=hist_type,
            align=align,
            label=label,
            color=color,
            alpha=alpha,
        )

        name = label if label is not None else self.__default_artist_label()
        self.artists[name] = Artist(axis, "hist", color, histogram)

        return None

    def imshow(self, data: Array, cmap: str = "GnBu") -> None:

        data = np.array(data)
        if data.ndim != 2:
            raise ValueError("Data must be 2D.")
        if data.shape[0] != data.shape[1]:
            raise ValueError("Data must be square.")

        image = self.axes["left"].imshow(data, cmap=cmap)
        name = self.__default_artist_label()
        self.artists[name] = Artist("left", "image", None, image)

        return None

    def colormap(
        self, x: Array, y: Array, z: Array, cmap: str = "GnBu"
    ) -> None:

        map = self.axes["left"].pcolormesh(x, y, z, cmap=cmap)
        name = self.__default_artist_label()
        self.artists[name] = Artist("left", "image", None, map)

        return None

    def patch(self, patch) -> None:

        patch = self.axes["left"].add_patch(patch)
        name = self.__default_artist_label()
        self.artists[name] = Artist("left", "patch", None, patch)

    def contour(
        self,
        x: Array,
        y: Array,
        z: Array,
        levels: Array,
        color: Optional[str] = None,
        cmap: str = "GnBu",
    ) -> None:

        contours = self.axes["left"].contour(
            x, y, z, levels=levels, colors=color, cmap=cmap
        )
        name = self.__default_artist_label()
        self.artists[name] = Artist("left", "contour", color, contours)

        return None

    def contourf(
        self,
        x: Array,
        y: Array,
        z: Array,
        levels: Array,
        color: Optional[str] = None,
        cmap: str = "GnBu",
    ) -> None:

        contours = self.axes["left"].contourf(
            x, y, z, levels=levels, colors=color, cmap=cmap
        )
        name = self.__default_artist_label()
        self.artists[name] = Artist("left", "cmap", color, contours)

        return None


class SingleAxis(BaseFigure):

    @property
    def ax(self) -> Axes:
        return self.axes["left"]

    @property
    def left(self) -> Axes:
        return self.axes["left"]


class DoubleAxis(BaseFigure):

    @property
    def left(self) -> Axes:
        return self.axes["left"]

    @property
    def right(self) -> Axes:
        return self.axes["right"]

    @property
    def parax(self) -> Axes:
        return self.axes["parasite"]

    def custom_configuration(self) -> None:

        __right = self.axes["left"].twinx()
        assert isinstance(__right, Axes)
        self.axes["right"] = __right

        return None

    def custom_postprocessing(self) -> None:

        # Add color indicator to label
        lines = self.axes["right"].get_lines()
        # if len(lines) > 1:
        #     raise ValueError("Don't plot more than one line in the right axis.")
        self.axes["right"].yaxis.label.set_color(lines[0].get_color())

        # Do not show ticks if requested
        if not self.setup.show_tick_labels_r:
            self.axes["right"].yaxis.set_major_formatter(ticker.NullFormatter())
            self.axes["right"].yaxis.set_minor_formatter(ticker.NullFormatter())

        return None


class ParasiteAxis(BaseFigure):

    @property
    def left(self) -> Axes:
        return self.axes["left"]

    @property
    def right(self) -> Axes:
        return self.axes["right"]

    @property
    def parax(self) -> Axes:
        return self.axes["parasite"]

    def custom_configuration(self) -> None:

        __right = self.axes["left"].twinx()
        assert isinstance(__right, Axes)
        self.axes["right"] = __right

        __parax = self.axes["left"].twinx()
        assert isinstance(__parax, Axes)
        self.axes["parasite"] = __parax
        self.axes["parasite"].spines.right.set_position(("axes", 1.2))

        return None

    def custom_postprocessing(self) -> None:

        # Add color indicator to label
        lines = self.axes["right"].get_lines()
        if len(lines) > 1:
            raise ValueError("Don't plot more than one line in the right axis.")
        self.axes["right"].yaxis.label.set_color(lines[-1].get_color())

        lines = self.axes["parasite"].get_lines()
        if len(lines) > 1:
            raise ValueError(
                "Don't plot more than one line in the parasite axis."
            )
        self.axes["parasite"].yaxis.label.set_color(lines[-1].get_color())

        # Do not show ticks if requested
        if not self.setup.show_tick_labels_r:
            self.axes["right"].yaxis.set_major_formatter(ticker.NullFormatter())
            self.axes["right"].yaxis.set_minor_formatter(ticker.NullFormatter())
        if not self.setup.show_tick_labels_p:
            self.axes["parasite"].yaxis.set_major_formatter(
                ticker.NullFormatter()
            )
            self.axes["parasite"].yaxis.set_minor_formatter(
                ticker.NullFormatter()
            )

        return None


class Legend(BaseFigure):

    def __init__(
        self, setup: PlotSetup = PlotSetup(), _figure: FigureLike | None = None
    ) -> None:

        setup.show_axes = False
        setup.legend_location = "center"
        setup.ylim = (1, 2)
        setup.xlim = (1, 2)

        super().__init__(setup, _figure)

        return None

    def add_legend(self, figure: BaseFigure) -> None:

        # Disable legend from figure
        figure.setup.legend = False

        for key, val in figure.artists.items():

            if key.startswith(self._default_prefix):
                continue

            self.line(0, 0, fmt="o", color=val.color, label=key)

        return None


class Plot3D(BaseFigure):

    def __init__(
        self,
        setup: PlotSetup = PlotSetup(),
        _figure: FigureLike | None = None,
    ) -> None:

        setup.minor_ticks_x = False
        setup.minor_ticks_y = False
        setup.minor_ticks_z = False
        setup.minor_ticks = False

        super().__init__(setup, _figure)

    def generate_subplot(self) -> Axes:

        ax = self.figure.add_subplot(
            projection="3d",
            proj_type=self.setup.projection,
            box_aspect=(1, 1, 1),
            azim=50,
        )
        return ax

    def custom_postprocessing(self) -> None:

        ax: Any = self.axes["left"]

        if ax.get_zscale() == "linear":
            ax.ticklabel_format(
                axis="z", scilimits=self.setup.scilimits_z, useMathText=True
            )
            if self.setup.minor_ticks_z:
                ax.zaxis.set_minor_locator(ticker.AutoMinorLocator())

        layout_engine = self.figure.get_layout_engine()
        assert layout_engine is not None
        layout_engine.set(w_pad=self.setup.w_padding, h_pad=self.setup.h_padding)  # type: ignore

    def line(
        self,
        x: Array | Scalar,
        y: Optional[Array | Scalar] = None,
        z: Optional[Array | Scalar] = None,
        fmt: str = "-",
        width: Optional[float] = None,
        markersize: Optional[float] = None,
        color: Optional[str] = None,
        alpha: float = 1.0,
        label: Optional[str] = None,
        axis: str = "left",
    ) -> None:

        assert y is not None and z is not None
        (line,) = self.axes[axis].plot(
            x,
            y,
            z,
            fmt,
            linewidth=width,
            markersize=markersize,
            color=color,
            alpha=alpha,
            label=label,
        )

        name = label if label is not None else self.__default_artist_label()
        self.artists[name] = Artist(axis, "line", color, line)

        return None

    def surface(
        self,
        x: Array,
        y: Array,
        z: Array,
        color: Optional[str] = None,
        alpha: float = 1.0,
        label: Optional[str] = None,
        axis: str = "left",
    ) -> None:

        if color is None:
            for _ in range(len(self.artists.keys())):
                self.next_color()
            color = self.next_color()
        surface = self.axes[axis].plot_surface(  # type: ignore
            x, y, z, color=color, alpha=alpha, label=label
        )

        name = label if label is not None else self.__default_artist_label()
        self.artists[name] = Artist(axis, "surface", color, surface)

        return None


class Mosaic(Canvas):

    def __init__(self, mosaic: str, setup: Optional[PlotSetup] = None) -> None:
        super().__init__(mosaic, setup if setup is not None else PlotSetup())

    def subplot(
        self,
        setup: Optional[PlotSetup] = None,
        generator: type[PlotType] = SingleAxis,
    ) -> PlotType:

        if setup is None:
            setup = PlotSetup()

        subfigure = self.canvas.add_subfigure(
            self.canvas_gridspec[next(self.canvas_structure)],
            facecolor=setup.subfigure_color,
            edgecolor=setup.subfigure_edgecolor,
            linewidth=setup.subfigure_edgewidth,
        )

        if setup.subfigure_title:
            subfigure.suptitle(setup.subfigure_title)

        return generator(setup, subfigure)
