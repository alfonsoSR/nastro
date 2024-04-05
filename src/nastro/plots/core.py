from dataclasses import dataclass
from nastro.types import Double
from typing import Self
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure as mplFigure
from matplotlib.lines import Line2D
from matplotlib.collections import PolyCollection
from matplotlib.container import ErrorbarContainer, BarContainer
from matplotlib.rcsetup import cycler
from typing import Sequence, TypeVar, Any, Literal
import numpy as np

# TODO: ADD DOCSTRINGS TO PUBLIC METHODS
# TODO: SUPPORT FOR 3D PLOTS
# TODO: SUPPORT FOR FANCY PLOTS (EX: POLAR, IMSHOW, ETC.)

type ArrayLike = np.ndarray | Sequence
SubplotType = TypeVar("SubplotType", bound="Plot")

color_cycler = cycler(
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
    """Plot configuration.

    :param figsize: Figure size in inches
    :param layout: Layout of the subplots.
        Possible values: "tight", "constrained", "none" or "compressed"
    :param title: Figure title
    :param show: Whether to show the plot or not
    :param save: Whether to save the plot or not
    :param dir: Directory to save plot
    :param name: Plot filename
    :param subtitle: Subplot title
    :param xlabel: x-axis label
    :param ylabel: y-axis label
    :param rlabel: Right axis label
    :param plabel: Parasite axis label
    :param xscale: x-axis scale
        Possible values: "linear", "log", "symlog", "logit"
    :param yscale: y-axis scale
        Possible values: "linear", "log", "symlog", "logit"
    :param rscale: Right axis scale
        Possible values: "linear", "log", "symlog", "logit"
    :param pscale: Parasite axis scale
        Possible values: "linear", "log", "symlog", "logit"
    :param xlim: x-axis limits
    :param ylim: y-axis limits
    :param rlim: Right axis limits
    :param plim: Parasite axis limits
    :param legend: Show legend
    :param legend_location: Legend location
        Possible values: "best", "upper right", "upper left", "lower left",
        "lower right", "right", "center left", "center right", "lower center",
        "upper center", "center"
    :param grid: Show grid
    """

    # Figure configuration
    figsize: tuple[float, float] = (7, 5)
    layout: str = "tight"
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
    rlabel: str | None = None
    plabel: str | None = None

    xscale: str = "linear"
    yscale: str = "linear"
    rscale: str = "linear"
    pscale: str = "linear"

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    rlim: tuple[float, float] | None = None
    plim: tuple[float, float] | None = None

    legend: bool = True
    legend_location: str = "best"

    grid: bool = False

    def __post_init__(self) -> None:

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

        self.fig = plt.figure(figsize=self.setup.figsize, layout=self.setup.layout)
        __subplots = self.fig.subplot_mosaic(mosaic)
        self.subplots = iter(__subplots.values())

        if self.setup.title is not None:
            self.fig.suptitle(self.setup.title)

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
            self.ax.grid()

        self.color_cycler = iter(color_cycler)
        self.axes_dict: dict[str, Axes] = {"left": self.ax}
        self.lines: dict[str, tuple[str, Line2D]] = {}
        self.boundaries: dict[Line2D, PolyCollection] = {}
        # self.error_bars: dict[str, tuple[str, ErrorbarContainer]] = {}
        self.artists: dict[str, tuple[str, Any]] = {}

        return None

    def add_line(
        self,
        x: ArrayLike,
        y: ArrayLike | None = None,
        fmt: str = "-",
        markersize: float | None = 3,
        color: str | None = None,
        label: str | None = None,
        axis: str = "left",
    ) -> None:

        if y is None:
            (line,) = self.axes_dict[axis].plot(
                x, fmt, label=label, color=color, markersize=markersize
            )
        else:
            (line,) = self.axes_dict[axis].plot(
                x, y, fmt, label=label, color=color, markersize=markersize
            )

        if label is not None:
            self.artists[label] = (axis, line)
            self.lines[label] = (axis, line)
        else:
            __number = len(self.lines) + 1
            self.artists[f"line{__number}"] = (axis, line)
            self.lines[f"line{__number}"] = (axis, line)

        return None

    def add_boundary(
        self,
        limit: ArrayLike | Double,
        line: str | None = None,
        follow_reference: bool = False,
        alpha: float = 0.2,
    ) -> None:

        if line is not None:
            if line not in self.lines.keys():
                raise ValueError(f"Line {line} not found in the plot")
            target_axis, target = self.lines[line]
        else:
            target_axis, target = list(self.lines.values())[-1]

        x = target.get_xdata()
        reference = target.get_ydata() if follow_reference else np.zeros_like(x)
        limit = np.array(limit)

        boundary = self.axes_dict[target_axis].fill_between(
            x, reference + limit, reference - limit, alpha=alpha
        )

        self.boundaries[target] = boundary

        return None

    def add_errorbar(
        self,
        x: ArrayLike,
        y: ArrayLike,
        error: ArrayLike,
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
        x: ArrayLike,
        y: ArrayLike | None = None,
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
        x: ArrayLike,
        height: ArrayLike,
        width: float = 0.8,
        ticks: ArrayLike | None = None,
        axis: str = "left",
    ) -> None:

        bar = self.axes_dict[axis].bar(x, height, width=width, tick_label=ticks)
        count = len(self.artists) + 1
        self.artists[f"artist{count}"] = (axis, bar)

        return None

    def add_horizontal_barplot(
        self,
        x: ArrayLike,
        width: ArrayLike,
        height: float = 0.8,
        ticks: ArrayLike | None = None,
        axis: str = "left",
    ) -> None:

        bar = self.axes_dict[axis].barh(x, width, height=height, tick_label=ticks)
        count = len(self.artists) + 1
        self.artists[f"artist{count}"] = (axis, bar)

        return None

    def next_color(self) -> str:
        try:
            return next(self.color_cycler)["color"]
        except StopIteration:
            self.color_cycler = iter(color_cycler)
            return next(self.color_cycler)["color"]

    def postprocess(self) -> None:

        # Colors
        for _, artist in self.artists.values():

            if isinstance(artist, Line2D):
                artist.set_color(self.next_color())

            elif isinstance(artist, ErrorbarContainer):
                color = self.next_color()
                artist[0].set_color(color)
                for tick in artist[2]:
                    tick.set_color(color)

            elif isinstance(artist, BarContainer):
                for bar in artist:
                    bar.set_color(self.next_color())

            elif isinstance(artist, list):
                if isinstance(artist[0], Line2D):
                    color = self.next_color()
                    for line in artist:
                        line.set_color(color)
                else:
                    raise ValueError(f"Unknown artist type")
            else:
                print(f"Unknown artist type: {artist.__name__}")

        for line, boundary in self.boundaries.items():
            boundary.set_color(line.get_color())

        # Legend
        __handles = []
        for axis in self.axes_dict.values():
            for handle in axis.get_legend_handles_labels()[0]:
                __handles.append(handle)
        if self.setup.legend and len(__handles) > 0:
            last_axis = list(self.axes_dict.values())[-1]
            last_axis.legend(loc=self.setup.legend_location, handles=__handles)

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

        plt.close()

        return None


class SingleAxis(Plot):
    pass


class DoubleAxis(Plot):

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

    def __init__(self, mosaic: str, setup: PlotSetup = PlotSetup()) -> None:
        super().__init__(setup, mosaic)

    def add_subplot(
        self,
        generator: type[SubplotType] = SingleAxis,
        setup: PlotSetup = PlotSetup(),
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

        plt.close()
        return None


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
#         self, x: Array, y: Array, z: Array, fmt: str = "-", label: str | None = None
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
#         self, x: Array, y: Array, z: Array, fmt: str = "-", label: str | None = None
#     ) -> str:
#         self.add_line(x, y, z, fmt=fmt, label=label)
#         self.postprocess()
#         return self.path
