from typing import (
    Any,
    TypeVar,
    TypeAlias,
    Sequence,
    Optional,
)
from matplotlib.figure import SubFigure, Figure as mplFigure
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import ticker
from ..types import Scalar, Array
from .settings import PlotSetup
from .core import BaseFigure, Artist, Canvas

PlotType = TypeVar("PlotType", bound="BaseFigure")
FigureLike: TypeAlias = SubFigure | mplFigure
AxesLike: TypeAlias = Axes | Axes3D
Plotable: TypeAlias = Scalar | Array | Sequence[Scalar]


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
