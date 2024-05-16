from dataclasses import dataclass
from .. import types as nt
from typing import Self
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure as mplFigure
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.container import ErrorbarContainer, BarContainer
from matplotlib.image import AxesImage
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3D
from matplotlib.rcsetup import cycler
from typing import Sequence, TypeVar, Any, Literal
import numpy as np

# TODO: ADD DOCSTRINGS TO PUBLIC METHODS
# TODO: SUPPORT FOR 3D PLOTS
# TODO: SUPPORT FOR FANCY PLOTS (EX: POLAR, IMSHOW, ETC.)

SubplotType = TypeVar("SubplotType", bound="Plot")

color_cycler = cycler(  # type: ignore
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
class PlotSetup:
    """Plot configuration

    PlotSetup is nastro's approach to plot configuration. It is a dataclass including
    most of the features one generally specifies when creating a plot in Matplotlib. This includes aspects such as figure size, labels, scales... but also flags to show or save the figure or to display a legend or a grid. The goal is to simplify the process of creating and customizing plots by concentrating all the configuration in a single object, which can then be used as input for all plotting classes.

    Parameters
    ------------
    figsize : tuple[float, float], optional
        Figure size.
    layout : Literal["tight", "constrained", "none", "compressed"], optional
        Layout used to calculate spacings and sizes of subplots.
    title : str, optional
        Figure title.
    show : bool, optional
        Whether to show the plot or not.
    save : bool, optional
        Whether to save the plot or not.
    dir : str | Path, optional
        Directory to save plot.
    name : str, optional
        Plot filename.
    path: Path, optional
        Full path to save the plot.
    axtitle : str, optional
        Subplot title.
    xlabel : str, optional
        x-axis label.
    ylabel : str, optional
        y-axis label.
    rlabel : str, optional
        Right axis label.
    plabel : str, optional
        Parasite axis label.
    xscale : Literal["linear", "log", "symlog", "logit"], optional
        x-axis scale.
    yscale : Literal["linear", "log", "symlog", "logit"], optional
        y-axis scale.
    rscale : Literal["linear", "log", "symlog", "logit"], optional
        Right axis scale.
    pscale : Literal["linear", "log", "symlog", "logit"], optional
        Parasite axis scale.
    xlim : tuple[float, float], optional
        x-axis limits.
    ylim : tuple[float, float], optional
        y-axis limits.
    rlim : tuple[float, float], optional
        Right axis limits.
    plim : tuple[float, float], optional
        Parasite axis limits.
    xticks : nt.Array, optional
        x-axis ticks.
    yticks : nt.Array, optional
        y-axis ticks.
    rticks : nt.Array, optional
        Right axis ticks.
    pticks : nt.Array, optional
        Parasite axis ticks.
    xticks_idx : nt.Array, optional
        x-axis ticks indices.
    yticks_idx : nt.Array, optional
        y-axis ticks indices.
    rticks_idx : nt.Array, optional
        Right axis ticks indices.
    pticks_idx : nt.Array, optional
        Parasite axis ticks indices.
    colorbar : bool, optional
        Show colorbar.
    cbar_label : str, optional
        Colorbar label.
    cbar_shrink : float, optional
        Colorbar shrink factor.
    legend : bool, optional
        Show legend.
    legend_location : str, optional
        Legend location. Possible values: "best", "upper right", "upper left", "lower left", "lower right", "right", "center left", "center right", "lower center", "upper center", "center".
    grid : bool, optional
        Show grid.
    grid_alpha : float, optional
        Grid transparency.
    """

    # Figure configuration
    figsize: tuple[float, float] = (7, 5)
    layout: Literal["tight", "constrained", "none", "compressed"] = "tight"
    aspect: Literal["auto", "equal"] = "auto"

    title: str | None = None
    show: bool = True
    save: bool = False
    dir: str | Path | None = None
    name: str | None = None
    path: Path | None = None

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

    xticks: nt.Array | None = None
    yticks: nt.Array | None = None
    zticks: nt.Array | None = None
    rticks: nt.Array | None = None
    pticks: nt.Array | None = None
    xticks_idx: nt.Array | None = None
    yticks_idx: nt.Array | None = None
    zticks_idx: nt.Array | None = None
    rticks_idx: nt.Array | None = None
    pticks_idx: nt.Array | None = None

    colorbar: bool = True
    cbar_label: str | None = None
    cbar_shrink: float = 1.0

    legend: bool = True
    legend_location: str = "best"
    legend_title: str | None = None
    legend_columns: int = 1

    grid: bool = False
    grid_alpha: float = 0.15

    def __post_init__(self) -> None:

        if self.xticks is not None and self.xticks_idx is None:
            self.xticks_idx = np.arange(len(self.xticks))
        if self.yticks is not None and self.yticks_idx is None:
            self.yticks_idx = np.arange(len(self.yticks))
        if self.zticks is not None and self.zticks_idx is None:
            self.zticks_idx = np.arange(len(self.zticks))
        if self.rticks is not None and self.rticks_idx is None:
            self.rticks_idx = np.arange(len(self.rticks))
        if self.pticks is not None and self.pticks_idx is None:
            self.pticks_idx = np.arange(len(self.pticks))

        if self.save:
            if self.name is None:
                raise ValueError("Missing a filename to save the plot")
            if self.dir is None:
                raise ValueError("Missing a directory to save the plot")

        if self.name is not None:
            if self.dir is None:
                raise ValueError("Missing a directory to save the plot")
            elif isinstance(self.dir, str):
                self.dir = Path(self.dir)

            self.path = self.dir / self.name

        return None

    def copy(self) -> "PlotSetup":
        return PlotSetup(**self.__dict__)


class BaseFigure:

    def __init__(self, setup: PlotSetup, mosaic: str = "a") -> None:

        self.setup = setup

        self.fig = plt.figure(  # type: ignore
            figsize=self.setup.figsize, layout=self.setup.layout
        )
        __subplots = self.fig.subplot_mosaic(mosaic)
        self.subplots = iter(__subplots.values())

        if self.setup.title is not None:
            self.fig.suptitle(self.setup.title)  # type: ignore

        return None


class Plot(BaseFigure):

    def __init__(
        self,
        setup: PlotSetup = PlotSetup(),
        _figure: mplFigure | None = None,
        _axes: Axes | None = None,
    ) -> None:

        if _figure is None and _axes is None:

            self.is_subplot = False
            super().__init__(setup)
            self.ax = next(self.subplots)

        elif _figure is not None and _axes is not None:
            self.is_subplot = True
            self.setup = setup
            self.fig = _figure
            self.ax = _axes
        else:
            raise ValueError("Pass both figure and axes or none of them")

        if self.setup.axtitle is not None:
            self.ax.set_title(self.setup.axtitle)

        if self.setup.xlabel is not None:
            self.ax.set_xlabel(self.setup.xlabel)
        if self.setup.ylabel is not None:
            self.ax.set_ylabel(self.setup.ylabel)

        self.ax.set_xscale(self.setup.xscale)
        self.ax.set_yscale(self.setup.yscale)

        if self.setup.xlim is not None:
            self.ax.set_xlim(self.setup.xlim)
        if self.setup.ylim is not None:
            self.ax.set_ylim(self.setup.ylim)

        if self.setup.grid:
            self.ax.grid(alpha=self.setup.grid_alpha, which="both")

        self.color_cycler = iter(color_cycler)
        self.axes_dict: dict[str, Axes] = {"left": self.ax}
        self.lines: dict[str, tuple[str, Line2D]] = {}
        # self.boundaries: dict[Line2D, PolyCollection] = {}
        # self.error_bars: dict[str, tuple[str, ErrorbarContainer]] = {}
        self.artists: dict[str, tuple[str, Any]] = {}

        return None

    def add_line(
        self,
        x: nt.Array,
        y: nt.Array | None = None,
        fmt: str = "-",
        width: float | None = None,
        markersize: float | None = None,
        color: str | None = None,
        alpha: float = 1.0,
        label: str | None = None,
        axis: str = "left",
    ) -> None:
        """Line

        :param x: Data for the x-axis.
        :param y: Data for the y-axis. If set to None, the data in x will be
            plotted against a range of integers with the same length.
        :param fmt: Line style (no color).
        :param width: Line width.
        :param markersize: Size of the markers.
        :param color: Line color as name or hex code.
        :param alpha: Transparency of the line.
        :param label: Label to identify the line in the legend.
        :param axis: Axis to plot the line.
        """

        if y is None:
            (line,) = self.axes_dict[axis].plot(
                x,
                fmt,
                label=label,
                linewidth=width,
                markersize=markersize,
                alpha=alpha,
            )
        else:
            (line,) = self.axes_dict[axis].plot(
                x,
                y,
                fmt,
                label=label,
                linewidth=width,
                markersize=markersize,
                alpha=alpha,
            )

        name = label if label is not None else f"line{len(self.lines) + 1}"
        self.lines[name] = (axis, line)
        self.artists[name] = (axis, (color, line))

        return None

    def add_scatter(
        self,
        x: nt.Array,
        y: nt.Array | None = None,
        fmt: str = ".",
        width: float | None = None,
        markersize: float | None = 2.0,
        color: str | None = None,
        alpha: float = 1.0,
        label: str | None = None,
        axis: str = "left",
    ) -> None:
        self.add_line(x, y, fmt, width, markersize, color, alpha, label, axis)
        return None

    def add_boundary(
        self,
        limits: nt.Array | nt.Double,
        line: str | Literal["last"] | None = "last",
        follow: bool = False,
        color: str | None = None,
        alpha: float = 0.1,
        label: str | None = None,
        axis: str = "left",
    ) -> None:
        """Solid boundary

        Creates a solid boundary by adding and subtracting a specified limit from
        a reference, which might be a line or just the horizontal axis.

        :param limits: A number or a sequence of numbers representing the limits
            of the boundary.
        :param line: The line to which the boundary will be attached. Can be set
            to "last" to follow the last line added to the plot, to a string
            representing the label of the line that is currently in the plot or
            to None to use the horizontal axis as reference.
        :param follow: Whether the boundary should follow the reference or not.
        :param color: Color of the boundary. If set to None, the boundary will
            have a semi-transparent version of the color of the reference.
        :param alpha: Transparency of the boundary.
        :param label: Label to identify the boundary in the legend.
        :param axis: Axis to plot the boundary.
        """

        if line is None:

            if nt.is_double(limits):
                raise ValueError(
                    "Failed to generate boundary. Missing information about x-axis"
                )

            elif nt.is_array(limits):
                target_axis = axis
                target = None
                limits = np.array(limits, dtype=np.float64)
                x = np.arange(len(limits))
                reference = np.zeros_like(x)

            else:
                raise ValueError(
                    "Failed to generate boundary."
                    " Limits must be a number or a sequence"
                )

        else:

            if line == "last":
                target_axis, target = list(self.lines.values())[-1]
            else:
                if line not in self.lines.keys():
                    raise ValueError(
                        "Failed to generate boundary. "
                        f"Line {line} not found in the plot"
                    )
                target_axis, target = self.lines[line]

            x = target.get_xdata()
            reference = target.get_ydata() if follow else np.zeros_like(x)

            if nt.is_array(limits):
                limits = np.array(limits)
                if limits.shape != np.array(x).shape:
                    raise ValueError(
                        "Failed to generate boundary."
                        " Limits and line have incompatible shapes."
                    )
            else:
                limits = np.array(limits)

        boundary = self.axes_dict[target_axis].fill_between(
            x,
            reference + limits,
            reference - limits,
            alpha=alpha,
            label=label,
        )

        out = (target_axis, (target, color, boundary))

        if label is not None:
            self.artists[label] = out
        else:
            count = len(self.artists) + 1
            self.artists[f"bound-{count}"] = out

        return None

    def add_vertical_boundary(
        self,
        low: nt.Double,
        high: nt.Double,
        color: str | None = None,
        alpha: float = 0.1,
        label: str | None = None,
        axis: str = "left",
    ) -> None:
        """Vertical boundary

        Generates a colored region between two vertical lines spanning the entire
        height of the plot.

        :param low: Lower limit of the boundary.
        :param high: Upper limit of the boundary.
        :param color: Color of the boundary.
        :param alpha: Transparency of the boundary.
        :param label: Label to identify the boundary in the legend.
        :param axis: Axis to plot the boundary.
        """

        boundary = self.axes_dict[axis].add_artist(
            Rectangle(
                (low, -1e20),  # type: ignore
                high - low,  # type: ignore
                2e20,
                alpha=alpha,
                color=color,
                label=label,
            )
        )

        out = (axis, (None, color, boundary))
        name = label if label is not None else f"vbound-{len(self.artists) + 1}"
        self.artists[name] = out

        return None

    def add_errorbar(
        self,
        x: nt.Array,
        y: nt.Array,
        error: nt.Array,
        fmt: str = "-",
        color: str | None = None,
        label: str | None = None,
        axis: str = "left",
    ) -> None:

        errorbar = self.axes_dict[axis].errorbar(
            x, y, yerr=error, fmt=fmt, label=label, color=color
        )

        if label is not None:
            # self.error_bars[label] = (axis, errorbar)
            self.artists[label] = (axis, errorbar)
        else:
            __number = len(self.artists) + 1
            # self.error_bars[f"errorbar{__number}"] = (axis, errorbar)
            self.artists[f"artist{__number}"] = (axis, errorbar)

        return None

    def add_step(
        self,
        x: nt.Array,
        y: nt.Array | None = None,
        where: Literal["pre", "post", "mid"] = "pre",
        fmt: str = "-",
        color: str | None = None,
        label: str | None = None,
        axis: str = "left",
    ) -> None:

        if y is None:
            step = self.axes_dict[axis].step(
                x, fmt, where=where, label=label, color=color
            )
        else:
            step = self.axes_dict[axis].step(
                x, y, fmt, where=where, label=label, color=color
            )

        if label is not None:
            self.artists[label] = (axis, step)
        else:
            __number = len(self.artists) + 1
            self.artists[f"artist{__number}"] = (axis, step)

        return None

    def add_barplot(
        self,
        x: nt.Array,
        height: nt.Array,
        width: float = 0.8,
        ticks: nt.Array | None = None,
        axis: str = "left",
    ) -> None:

        bar = self.axes_dict[axis].bar(x, height, width=width, tick_label=ticks)
        count = len(self.artists) + 1
        self.artists[f"bar-{count}"] = (axis, bar)

        return None

    def add_horizontal_barplot(
        self,
        x: nt.Array,
        width: nt.Array,
        height: float = 0.8,
        ticks: nt.Array | None = None,
        axis: str = "left",
    ) -> None:

        bar = self.axes_dict[axis].barh(x, width, height=height, tick_label=ticks)
        count = len(self.artists) + 1
        self.artists[f"hbar-{count}"] = (axis, bar)

        return None

    def add_histogram(
        self,
        data: nt.Array,
        bins: int = 10,
        normalize: bool = False,
        cumulative: bool = False,
        hist_type: Literal["bar", "step", "stepfilled"] = "bar",
        align: Literal["left", "mid", "right"] = "mid",
        label: str | None = None,
        alpha: float = 0.8,
        axis: str = "left",
    ) -> None:
        """Histogram

        :param data: A sequence of numbers.
        :param bins: Number of intervals in which the data is divided.
        :param normalize: If False, the vertical axis represents the number of
            elements in each bin. If True, the counts are normalized by the total
            number of elements, so that the vertical axis represents the
            probability of a number belonging to each interval.
        :param cumulative: If True, each bin contains the number of elements that
            are smaller or equal to its upper limit.
        :param hist_type: Bar plots a sequence of rectangles, step plots the line
            joining the top of the rectangles, stepfilled plots the same line as
            step but fills the area below it.
        :param align: Whether bars will be centered on the left edge, right edge
            or in the middle of the intervals.
        :param label: To be shown in the legend to identify the histogram.
        :param alpha: Transparency of the bars.
        :param axis: Axis to plot the histogram.
        """

        _, _, hist = self.axes_dict[axis].hist(
            data,
            bins=bins,
            density=normalize,
            cumulative=cumulative,
            histtype=hist_type,
            align=align,
            label=label,
            alpha=alpha,
        )

        count = len(self.artists) + 1
        self.artists[f"hist-{count}"] = (axis, hist)

        return None

    def add_image(self, data: nt.Vector, cmap: str = "GnBu") -> None:

        if len(data.shape) != 2:
            raise ValueError("Data must be a 2D array")

        if data.shape[0] != data.shape[1]:
            raise ValueError("Data must be a square matrix")

        image = self.axes_dict["left"].imshow(data, cmap=cmap)
        __number = len(self.artists) + 1
        self.artists[f"artist{__number}"] = ("left", image)

        return None

    def next_color(self) -> str:
        try:
            return next(self.color_cycler)["color"]
        except StopIteration:
            self.color_cycler = iter(color_cycler)
            return next(self.color_cycler)["color"]

    def postprocess(self) -> None:

        # TODO: REFACTOR COLORING

        # Ticks
        if self.setup.xticks is not None and self.setup.xticks_idx is not None:
            self.ax.set_xticks(self.setup.xticks_idx, self.setup.xticks)
        if self.setup.yticks is not None and self.setup.yticks_idx is not None:
            self.ax.set_yticks(self.setup.yticks_idx, self.setup.yticks)
        if "right" in self.axes_dict.keys():
            if self.setup.rticks is not None and self.setup.rticks_idx is not None:
                self.axes_dict["right"].set_yticks(
                    self.setup.rticks_idx, self.setup.rticks
                )
        if "parasite" in self.axes_dict.keys():
            if self.setup.pticks is not None and self.setup.pticks_idx is not None:
                self.axes_dict["parasite"].set_yticks(
                    self.setup.pticks_idx, self.setup.pticks
                )

        # Colors
        for key, (_, artist) in self.artists.items():

            # Line
            if isinstance(artist, tuple) and isinstance(artist[-1], Line2D):
                color, line = artist
                if color is None:
                    color = self.next_color()
                line.set_color(color)

            # if isinstance(artist, Line2D):
            #     artist.set_color(self.next_color())

            elif isinstance(artist, ErrorbarContainer):
                color = self.next_color()
                artist[0].set_color(color)
                for tick in artist[2]:
                    tick.set_color(color)

            elif isinstance(artist, BarContainer):

                if "hist" in key:
                    color = self.next_color()
                    for bar in artist:
                        bar.set_color(color)
                else:
                    for bar in artist:
                        bar.set_color(self.next_color())

            # Vertical boundary
            elif isinstance(artist, tuple) and isinstance(artist[2], Rectangle):
                target, color, boundary = artist
                if color is None:
                    color = self.next_color() if target is None else target.get_color()
                boundary.set_color(color)

            elif isinstance(artist, list):
                if isinstance(artist[0], Line2D):
                    color = self.next_color()
                    for line in artist:
                        line.set_color(color)
                else:
                    try:
                        color = self.next_color()
                        for item in artist:
                            item.set_color(color)
                    except:
                        raise ValueError(f"Failed to color artist: {type(artist[0])}")

            elif isinstance(artist, AxesImage):

                if not self.setup.colorbar:
                    continue

                self.fig.colorbar(
                    artist,
                    ax=self.ax,
                    label=self.setup.cbar_label,
                    shrink=self.setup.cbar_shrink,
                )

            # Boundary
            elif isinstance(artist, tuple) and isinstance(artist[-1], PolyCollection):
                target, color, boundary = artist
                if color is None:
                    color = self.next_color() if target is None else target.get_color()
                boundary.set_color(color)
            else:
                print(f"Unknown artist type: {type(artist)}")

        # Legend
        __handles = []
        for axis in self.axes_dict.values():
            for handle in axis.get_legend_handles_labels()[0]:
                __handles.append(handle)
        if self.setup.legend and len(__handles) > 0:
            last_axis = list(self.axes_dict.values())[-1]
            last_axis.legend(
                loc=self.setup.legend_location,
                handles=__handles,
                title=self.setup.legend_title,
                ncols=self.setup.legend_columns,
            )

        # Aspect ratio
        for axis in self.axes_dict.values():
            axis.set_aspect(self.setup.aspect)

        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:

        self.postprocess()

        if "right" in self.axes_dict.keys():
            if len(self.axes_dict["right"].get_lines()) > 1:
                raise ValueError("Attempted to plot multiple lines on the right axis")
            line = self.axes_dict["right"].get_lines()[-1]
            self.axes_dict["right"].yaxis.label.set_color(line.get_color())

        if "parasite" in self.axes_dict.keys():
            if len(self.axes_dict["parasite"].get_lines()) > 1:
                raise ValueError(
                    "Attempted to plot multiple lines on the parasite axis"
                )
            line = self.axes_dict["parasite"].get_lines()[-1]
            self.axes_dict["parasite"].yaxis.label.set_color(line.get_color())

        if self.is_subplot:
            return None

        if self.setup.save:
            assert isinstance(self.setup.path, Path)
            self.setup.path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(self.setup.path)

        if self.setup.show:
            plt.show(block=True)

        plt.close("all")

        return None


class SingleAxis(Plot):
    """A 2D plot with a single vertical axis"""

    pass


class DoubleAxis(Plot):
    """A 2D plot with left and right vertical axes"""

    def __init__(
        self,
        setup: PlotSetup = PlotSetup(),
        _figure: mplFigure | None = None,
        _axes: Axes | None = None,
    ) -> None:

        super().__init__(setup, _figure, _axes)

        # Add right axis
        self.right = self.ax.twinx()
        assert isinstance(self.right, Axes)
        self.axes_dict["right"] = self.right

        if self.setup.rlabel is not None:
            self.right.set_ylabel(self.setup.rlabel)
        self.right.set_yscale(self.setup.rscale)
        if self.setup.rlim is not None:
            self.right.set_ylim(self.setup.rlim)

        return None


class ParasiteAxis(DoubleAxis):
    """A 2D plot with left, right and parasite vertical axes"""

    def __init__(
        self,
        setup: PlotSetup = PlotSetup(),
        _figure: mplFigure | None = None,
        _axes: Axes | None = None,
    ) -> None:

        super().__init__(setup, _figure, _axes)

        # Add parasite axis
        self.parax = self.ax.twinx()
        assert isinstance(self.parax, Axes)
        self.parax.spines.right.set_position(("axes", 1.13))
        self.axes_dict["parasite"] = self.parax

        if self.setup.plabel is not None:
            self.parax.set_ylabel(self.setup.plabel)
        self.parax.set_yscale(self.setup.pscale)
        if self.setup.plim is not None:
            self.parax.set_ylim(self.setup.plim)

        return None


class Mosaic(BaseFigure):
    """A figure with multiple subplots arranged in a grid"""

    def __init__(self, mosaic: str, setup: PlotSetup = PlotSetup()) -> None:
        super().__init__(setup, mosaic)

    def add_subplot(
        self,
        setup: PlotSetup = PlotSetup(),
        generator: type[SubplotType] = SingleAxis,
    ) -> SubplotType:
        return generator(setup, self.fig, next(self.subplots))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:

        if exc_type is not None:
            plt.close(self.fig)
            raise exc_type(exc_value).with_traceback(exc_traceback)

        if self.setup.save:
            assert isinstance(self.setup.path, Path)
            self.setup.path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(self.setup.path)

        if self.setup.show:
            plt.show(block=True)

        plt.close("all")
        return None


class Base3D:

    def __init__(self, setup: PlotSetup = PlotSetup()) -> None:

        self.setup = setup

        # Create figure and axis
        self.fig = plt.figure(figsize=self.setup.figsize)
        self.ax = self.fig.add_subplot(
            projection="3d", proj_type="ortho", box_aspect=(1, 1, 1)
        )

        # Set title and labels
        if self.setup.title is not None:
            self.fig.suptitle(self.setup.title)

        if self.setup.xlabel is not None:
            self.ax.set_xlabel(self.setup.xlabel)
        if self.setup.xscale is not None:
            self.ax.set_xscale(self.setup.xscale)
        if self.setup.xlim is not None:
            self.ax.set_xlim(self.setup.xlim)

        if self.setup.ylabel is not None:
            self.ax.set_ylabel(self.setup.ylabel)
        if self.setup.yscale is not None:
            self.ax.set_yscale(self.setup.yscale)
        if self.setup.ylim is not None:
            self.ax.set_ylim(self.setup.ylim)

        if self.setup.zlabel is not None:
            self.ax.set_zlabel(self.setup.zlabel)  # type: ignore
        if self.setup.zscale is not None:
            self.ax.set_zscale(self.setup.zscale)  # type: ignore
        if self.setup.zlim is not None:
            self.ax.set_zlim(self.setup.zlim)  # type: ignore

        self.color_cycler = iter(color_cycler)
        self.axes_dict: dict[str, Axes] = {"left": self.ax}
        self.artists: dict[str, tuple[str, Any]] = {}

        return None

    def next_color(self) -> str:
        try:
            return next(self.color_cycler)["color"]
        except StopIteration:
            self.color_cycler = iter(color_cycler)
            return next(self.color_cycler)["color"]

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        self.postprocess()

    def add_line(
        self,
        x: nt.Array,
        y: nt.Array,
        z: nt.Array,
        fmt: str = "-",
        width: float | None = None,
        markersize: float | None = None,
        color: str | None = None,
        alpha: float = 1.0,
        label: str | None = None,
        axis: str = "left",
    ) -> None:
        """Line

        :param x: Data for the x-axis.
        :param y: Data for the y-axis. If set to None, the data in x will be
            plotted against a range of integers with the same length.
        :param fmt: Line style (no color).
        :param width: Line width.
        :param markersize: Size of the markers.
        :param color: Line color as name or hex code.
        :param alpha: Transparency of the line.
        :param label: Label to identify the line in the legend.
        """

        (line,) = self.axes_dict[axis].plot(
            x,
            y,
            z,
            fmt,
            label=label,
            linewidth=width,
            markersize=markersize,
            alpha=alpha,
        )

        name = label if label is not None else f"line{len(self.artists) + 1}"
        self.artists[name] = (axis, (color, line))

        return None

    def postprocess(self) -> None:

        original_limits = np.array(
            [
                self.ax.get_xlim(),
                self.ax.get_ylim(),
                self.ax.get_zlim(),  # type: ignore
            ]
        ).T

        homogeneous_limits = (np.min(original_limits[0]), np.max(original_limits[1]))

        self.ax.set_xlim(homogeneous_limits)
        self.ax.set_ylim(homogeneous_limits)
        self.ax.set_zlim(homogeneous_limits)  # type: ignore

        # Ticks
        if self.setup.xticks is not None and self.setup.xticks_idx is not None:
            self.ax.set_xticks(self.setup.xticks_idx, self.setup.xticks)
        if self.setup.yticks is not None and self.setup.yticks_idx is not None:
            self.ax.set_yticks(self.setup.yticks_idx, self.setup.yticks)

        # Colors
        for key, (_, artist) in self.artists.items():

            # Line
            if isinstance(artist, tuple) and isinstance(artist[-1], Line3D):
                color, line = artist
                if color is None:
                    color = self.next_color()
                line.set_color(color)
            else:
                print(f"Unknown artist type: {type(artist)}")

        # Legend
        __handles = []
        for axis in self.axes_dict.values():
            for handle in axis.get_legend_handles_labels()[0]:
                __handles.append(handle)
        if self.setup.legend and len(__handles) > 0:
            last_axis = list(self.axes_dict.values())[-1]
            last_axis.legend(
                loc=self.setup.legend_location,
                handles=__handles,
                title=self.setup.legend_title,
                ncols=self.setup.legend_columns,
            )

        if self.setup.save:
            assert isinstance(self.setup.path, Path)
            self.setup.path.parent.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(self.setup.path)

        if self.setup.show:
            plt.show(block=True)

        plt.close("all")

        return None


class Plot3D(Base3D):
    pass


# class Base3D:
#     def __init__(self, setup: PlotSetup = PlotSetup()) -> None:
#         self.setup = setup

#         # Create figure and axis
#         self.fig = plt.figure(figsize=self.setup.figsize)
#         self.ax = self.fig.add_subplot(
#             projection="3d", proj_type="ortho", box_aspect=(1, 1, 1)
#         )

#         # Set title and labels
#         if self.setup.title is not None:
#             self.fig.suptitle(self.setup.title, fontsize="x-large")

#         if self.setup.xlabel is not None:
#             self.ax.set_xlabel(self.setup.xlabel)
#         if self.setup.xscale is not None:
#             self.ax.set_xscale(self.setup.xscale)
#         if self.setup.xlim is not None:
#             self.ax.set_xlim(self.setup.xlim)

#         if self.setup.ylabel is not None:
#             self.ax.set_ylabel(self.setup.ylabel)
#         if self.setup.yscale is not None:
#             self.ax.set_yscale(self.setup.yscale)
#         if self.setup.ylim is not None:
#             self.ax.set_ylim(self.setup.ylim)

#         if self.setup.zlabel is not None:
#             self.ax.set_zlabel(self.setup.zlabel)  # type: ignore
#         if self.setup.zscale is not None:
#             self.ax.set_zscale(self.setup.zscale)  # type: ignore
#         if self.setup.zlim is not None:
#             self.ax.set_zlim(self.setup.zlim)  # type: ignore

#         self.path = ""

#         return None

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, exc_traceback):
#         self.postprocess()

#     def add_line(
#         self, x: nt.Array, y: Array, z: Array, fmt: str = "-", label: str | None = None
#     ) -> None:
#         self.ax.plot(x, y, z, fmt, label=label)

#         return None

#     def postprocess(self) -> str:
#         original_limits = np.array(
#             [
#                 self.ax.get_xlim(),
#                 self.ax.get_ylim(),
#                 self.ax.get_zlim(),  # type: ignore
#             ]
#         ).T

#         homogeneous_limits = (np.min(original_limits[0]), np.max(original_limits[1]))

#         self.ax.set_xlim(homogeneous_limits)
#         self.ax.set_ylim(homogeneous_limits)
#         self.ax.set_zlim(homogeneous_limits)  # type: ignore

#         labels = self.ax.get_legend_handles_labels()[1]
#         if self.setup.legend and len(labels) > 0:
#             self.ax.legend()

#         if self.setup.save:
#             Path(self.setup.dir).mkdir(parents=True, exist_ok=True)
#             self.path = f"{self.setup.dir}/{self.setup.name}"
#             self.fig.savefig(self.path)

#         if self.setup.show:
#             plt.show(block=True)
#             plt.close(self.fig)

#         return self.path


# class Plot3D(Base3D):
#     def plot(
#         self, x: nt.Array, y: Array, z: Array, fmt: str = "-", label: str | None = None
#     ) -> str:
#         self.add_line(x, y, z, fmt=fmt, label=label)
#         self.postprocess()
#         return self.path
