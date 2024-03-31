from dataclasses import dataclass, field
from nastro.types import Array, Double
from typing import Self
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure as mplFigure
from matplotlib.rcsetup import cycler
from typing import TypeVar, Sequence, Any
import numpy as np


SubplotType = TypeVar("SubplotType", bound="Subplot")
type ArrayLike = np.ndarray | Sequence

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

    :param figsize: Figure size in inches [(7, 5)]
    :param rows: Number of rows in the figure [1]
    :param cols: Number of columns in the figure [1]
    :param title: Figure title [None]
    :param axtitle: Axes title [None]
    :param xlabel: x-axis label [None]
    :param ylabel: y-axis label [None]
    :param rlabel: Right axis label [None]
    :param plabel: Parasite axis label [None]
    :param zlabel: Vertical axis label (3D) [None]
    :param legend: Show legend [True]
    :param xscale: x-axis scale [None]
    :param yscale: y-axis scale [None]
    :param rscale: Right axis scale [None]
    :param pscale: Parasite axis scale [None]
    :param zscale: Vertical axis scale (3D) [None]
    :param xlim: x-axis limits [None]
    :param ylim: y-axis limits [None]
    :param rlim: Right axis limits [None]
    :param plim: Parasite axis limits [None]
    :param zlim: Vertical axis limits (3D) [None]
    :param grid: Show grid [False]
    :param show: Show plot [True]
    :param save: Save plot [False]
    :param dir: Directory to save plot ["plots"]
    :param name: Plot filename [""]
    """

    # Figure configuration
    figsize: tuple[float, float] = (7, 5)
    rows: int = 1
    cols: int = 1

    # Title and labels
    title: str | None = None
    axtitle: str | None = None
    xlabel: str | None = None
    ylabel: str | None = None
    rlabel: str | None = None
    plabel: str | None = None
    zlabel: str | None = None
    legend: bool = True

    # Scales
    xscale: str | None = None
    yscale: str | None = None
    rscale: str | None = None
    pscale: str | None = None
    zscale: str | None = None

    # Limits
    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    rlim: tuple[float, float] | None = None
    plim: tuple[float, float] | None = None
    zlim: tuple[float, float] | None = None

    # Aesthetics
    grid: bool = False

    # Save and show configuration
    show: bool = True
    save: bool = False
    dir: str = "plots"
    name: str = ""
    path: Path = field(init=False)

    def __post_init__(self) -> None:
        self.path = Path(self.dir) / self.name

        return None

    def copy(self) -> "PlotSetup":
        return PlotSetup(**self.__dict__)


@dataclass
class FigureSetup:

    size: tuple[float, float] = (7, 5)
    grid: str = "a"
    layout: str = "tight"
    title: str | None = None
    show: bool = True
    save: bool = False
    dir: str = "plots"
    filename: str = ""

    __dir: Path = field(init=False)

    def __post_init__(self) -> None:

        self.__dir = Path(self.dir)
        return None


@dataclass
class SubplotSetup:

    title: str | None = None

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

    grid: bool = False
    legend: bool = True
    legend_loc: str = "best"


class Figure:

    def __init__(self, setup: FigureSetup = FigureSetup()) -> None:

        self.setup = setup
        self.fig = plt.figure(figsize=self.setup.size, layout=self.setup.layout)
        __subplots = self.fig.subplot_mosaic(self.setup.grid)
        self.subplots = iter(__subplots.values())

        if self.setup.title is not None:
            self.fig.suptitle(self.setup.title)

        return None

    def add_subplot(
        self, generator: type[SubplotType], setup: SubplotSetup = SubplotSetup()
    ) -> SubplotType:

        return generator(setup, self.fig, next(self.subplots))

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:

        if self.setup.save:
            self.setup.__dir.mkdir(parents=True, exist_ok=True)
            self.fig.savefig(self.setup.__dir / self.setup.filename)

        if self.setup.show:
            plt.show(block=True)

        plt.close(self.fig)
        return None


class Subplot:

    def __init__(self, setup: SubplotSetup, figure: mplFigure, axes: Axes) -> None:

        self.setup = setup
        self.fig = figure
        self.ax = axes
        self.axes_dict: dict[str, Any] = {"left": self.ax}
        self.color_cycler = iter(color_cycler)

        if self.setup.title is not None:
            self.ax.set_title(self.setup.title)

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

        return None

    def add_line(
        self,
        x: Array,
        y: Array | None = None,
        fmt: str = "-",
        label: str | None = None,
        axis: str = "left",
    ) -> None:

        if y is None:
            self.axes_dict[axis].plot(x, fmt, label=label)
        else:
            self.axes_dict[axis].plot(x, y, fmt, label=label)

        return None

    def add_boundary(
        self,
        limit: Array | Double,
        follow_reference: bool = False,
        alpha: float = 0.2,
        axis: str = "left",
    ) -> None:

        line = self.axes_dict[axis].get_lines()[-1]
        x = line.get_xdata()

        if isinstance(limit, Double):
            limit = limit * np.ones_like(x)

        reference = line.get_ydata() if follow_reference else np.zeros_like(x)

        self.axes_dict[axis].fill_between(
            x, reference + limit, reference - limit, alpha=alpha, color=line.get_color()
        )

        return None

    def postprocess(self) -> None:

        for line in self.ax.get_lines():
            line.set_color(next(self.color_cycler)["color"])

        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback) -> None:

        self.postprocess()

        __handles = []
        for axis in self.axes_dict.values():
            for line in axis.get_lines():
                __handles.append(line)

        if self.setup.legend and len(__handles) > 0:

            last_axis = list(self.axes_dict.values())[-1]
            last_axis.legend(loc=self.setup.legend_loc, handles=__handles)

        return None


# class BasePlot:
#     def __init__(self, setup: PlotSetup = PlotSetup(), _fig=None, _ax=None) -> None:
#         self.COLOR_CYCLE = iter(
#             [
#                 "#1f77b4",
#                 "#aec7e8",
#                 "#ff7f0e",
#                 "#ffbb78",
#                 "#2ca02c",
#                 "#98df8a",
#                 "#d62728",
#                 "#ff9896",
#                 "#9467bd",
#                 "#c5b0d5",
#                 "#8c564b",
#                 "#c49c94",
#                 "#e377c2",
#                 "#f7b6d2",
#                 "#7f7f7f",
#                 "#c7c7c7",
#                 "#bcbd22",
#                 "#dbdb8d",
#                 "#17becf",
#                 "#9edae5",
#             ]
#         )

#         self.setup: PlotSetup = setup

#         # Create figure and left axis
#         if _fig is None and _ax is None:
#             self.fig, self.ax = plt.subplots(
#                 figsize=self.setup.figsize,
#                 layout="tight",
#             )
#         elif _fig is not None and _ax is not None:
#             self.fig = _fig
#             self.ax = _ax
#         else:
#             raise ValueError("Provide both figure and axis or none of them")

#         # Set title and labels
#         if self.setup.title is not None:
#             self.fig.suptitle(
#                 self.setup.title,
#                 fontsize="x-large",
#             )

#         if self.setup.axtitle is not None:
#             self.ax.set_title(self.setup.axtitle)

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

#         self.path: str = ""

#         return None

#     def _formatter(self, x, pos):
#         if x == 0.0:
#             return f"{x:.0f}"
#         elif (np.abs(x) > 0.01) and (np.abs(x) < 1e2):
#             return f"{x:.1f}"
#         else:
#             a, b = f"{x:.0e}".split("e")
#             bsign = "-" if a[0] == "-" else ""
#             esign = "-" if b[0] == "-" else ""
#             exp = int(b[1:])
#             n = int(a[0]) if bsign == "" else int(a[1])
#             return f"{bsign}{n}e{esign}{exp}"
#             return f"${bsign}{n}" + r"\cdot 10^{" + f"{esign}{exp}" + r"}$"

#     def _postprocess(self) -> None:
#         for line in self.ax.get_lines():
#             line.set_color(next(self.COLOR_CYCLE))

#         return None

#     def postprocess(self) -> None:
#         self._postprocess()

#         # self.ax.yaxis.set_major_formatter(self._formatter)

#         labels = self.ax.get_legend_handles_labels()[1]
#         if self.setup.legend and len(labels) > 0:
#             self.ax.legend()

#         if self.setup.grid:
#             self.ax.grid()

#         if self.setup.save:
#             Path(self.setup.dir).mkdir(parents=True, exist_ok=True)
#             self.path = f"{self.setup.dir}/{self.setup.name}"
#             self.fig.savefig(self.path)

#         if self.setup.show:
#             plt.show(block=True)
#             plt.close(self.fig)

#         return None

#     def _plot(
#         self,
#         x: Array,
#         y: Array,
#         fmt: str | None = "-",
#         label: str | None = None,
#         axis: str | None = None,
#     ) -> None:
#         raise NotImplementedError

#     def add_line(
#         self,
#         x: Array,
#         y: Array,
#         fmt: str | None = "-",
#         label: str | None = None,
#         axis: str | None = "left",
#     ) -> None:
#         self._plot(x, y, fmt, label=label, axis=axis)
#         return None

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, exc_traceback):
#         self.postprocess()

#     def __call__(self) -> str:
#         self.postprocess()
#         return self.path


# class SinglePlot(BasePlot):
#     def _plot(
#         self,
#         x: Array,
#         y: Array,
#         fmt: str | None = "-",
#         label: str | None = None,
#         axis: str | None = None,
#     ) -> None:
#         self.ax.plot(x, y, fmt, label=label)
#         return None

#     def plot(
#         self, x: Array, y: Array, fmt: str | None = "-", label: str | None = None
#     ) -> str:
#         self._plot(x, y, fmt=fmt, label=label)
#         return self.__call__()


# class DoubleAxis(BasePlot):
#     def __init__(self, setup: PlotSetup = PlotSetup(), _fig=None, _ax=None) -> None:
#         setup.right_axis = True

#         super().__init__(setup, _fig, _ax)

#         self.left = self.ax
#         self.right = self.left.twinx()

#         if self.setup.rlabel is not None:
#             self.right.set_ylabel(self.setup.rlabel)
#         if self.setup.rscale is not None:
#             self.right.set_yscale(self.setup.rscale)
#         if self.setup.rlim is not None:
#             self.right.set_ylim(self.setup.rlim)

#         return None

#     def _postprocess(self) -> None:
#         super()._postprocess()

#         for line in self.right.get_lines():
#             line.set_color(next(self.COLOR_CYCLE))
#         right_line = self.right.get_lines()[0]
#         self.right.yaxis.label.set_color(right_line.get_color())
#         self.fig.subplots_adjust(left=0.1, right=0.8)
#         self.right.yaxis.set_major_formatter(self._formatter)

#         return None

#     def _plot(
#         self,
#         x: Array,
#         y: Array,
#         fmt: str | None = "-",
#         label: str | None = None,
#         axis: str | None = None,
#     ) -> None:
#         if axis == "left":
#             self.ax.plot(x, y, fmt, label=label)
#         elif axis == "right":
#             self.right.plot(x, y, fmt, label=label)
#         else:
#             raise ValueError("Axis must be either 'left' or 'right'")
#         return None

#     def plot(self, x: Vector, y_left: Vector, y_right: Vector) -> str:
#         self._plot(x, y_left, axis="left")
#         self._plot(x, y_right, axis="right")
#         return self.__call__()


# class ParasiteAxis(BasePlot):
#     def __init__(self, setup: PlotSetup = PlotSetup(), _fig=None, _ax=None) -> None:
#         setup.right_axis = True
#         setup.parasite_axis = True

#         super().__init__(setup, _fig, _ax)

#         self.left = self.ax
#         self.right = self.left.twinx()

#         if self.setup.rlabel is not None:
#             self.right.set_ylabel(self.setup.rlabel)
#         if self.setup.rscale is not None:
#             self.right.set_yscale(self.setup.rscale)
#         if self.setup.rlim is not None:
#             self.right.set_ylim(self.setup.rlim)

#         self.parax = self.left.twinx()
#         self.parax.spines.right.set_position(("axes", 1.13))

#         if self.setup.plabel is not None:
#             self.parax.set_ylabel(self.setup.plabel)
#         if self.setup.pscale is not None:
#             self.parax.set_yscale(self.setup.pscale)
#         if self.setup.plim is not None:
#             self.parax.set_ylim(self.setup.plim)

#         return None

#     def _postprocess(self) -> None:
#         super()._postprocess()

#         for line in self.right.get_lines():
#             line.set_color(next(self.COLOR_CYCLE))
#         right_line = self.right.get_lines()[0]
#         self.right.yaxis.label.set_color(right_line.get_color())
#         self.fig.subplots_adjust(left=0.1, right=0.8)
#         self.right.yaxis.set_major_formatter(self._formatter)

#         for line in self.parax.get_lines():
#             line.set_color(next(self.COLOR_CYCLE))
#         parax_line = self.parax.get_lines()[0]
#         self.parax.yaxis.label.set_color(parax_line.get_color())
#         self.parax.yaxis.set_major_formatter(self._formatter)

#         return None

#     def _plot(
#         self,
#         x: Array,
#         y: Array,
#         fmt: str | None = "-",
#         label: str | None = None,
#         axis: str | None = None,
#     ) -> None:
#         if axis == "left":
#             self.ax.plot(x, y, fmt, label=label)
#         elif axis == "right":
#             self.right.plot(x, y, fmt, label=label)
#         elif axis == "parax":
#             self.parax.plot(x, y, fmt, label=label)
#         else:
#             raise ValueError("Axis must be either 'left', 'right' or 'parax'")
#         return None

#     def plot(self, x: Vector, y_left: Vector, y_right: Vector, y_parax: Vector) -> str:
#         self._plot(x, y_left, axis="left")
#         self._plot(x, y_right, axis="right")
#         self._plot(x, y_parax, axis="parax")
#         return self.__call__()


# class MultiPlot:
#     def __init__(self, setup: PlotSetup) -> None:
#         self.setup = setup

#         if self.setup.subplots == (1, 1):
#             raise ValueError("Requesting a single plot with MultiPlot")

#         self.rows, self.cols = self.setup.subplots

#         self.fig, self.axes = plt.subplots(
#             self.rows, self.cols, figsize=self.setup.figsize, layout="tight"
#         )

#         self.ax_list = iter(self.axes.ravel())

#         if self.setup.title is not None:
#             self.fig.suptitle(self.setup.title, fontsize="x-large")

#         self.path = ""

#         return None

#     def __enter__(self):
#         return self

#     def __exit__(self, exc_type, exc_value, exc_traceback) -> None:
#         self.postprocess()
#         return None

#     def add_plot(self, setup: PlotSetup = PlotSetup(), type: type[T] = SinglePlot) -> T:
#         setup.show = False
#         setup.save = False
#         setup.legend = True if self.setup.legend else False
#         return type(setup, _fig=self.fig, _ax=next(self.ax_list))

#     def postprocess(self) -> str:
#         if self.setup.save:
#             Path(self.setup.dir).mkdir(parents=True, exist_ok=True)
#             self.path = f"{self.setup.dir}/{self.setup.name}"
#             self.fig.savefig(self.path)
#             if not self.setup.show:
#                 plt.close(self.fig)

#         if self.setup.show:
#             plt.show(block=True)
#             plt.close(self.fig)

#         return self.path

#     def __call__(self) -> str:
#         self.postprocess()
#         return self.path


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
