from .core import Subplot, SubplotSetup, mplFigure, Axes


class SingleAxis(Subplot):

    def __init__(self, setup: SubplotSetup, figure: mplFigure, axes: Axes) -> None:

        super().__init__(setup, figure, axes)

        return None


class DoubleAxis(Subplot):

    def __init__(self, setup: SubplotSetup, figure: mplFigure, axes: Axes) -> None:

        super().__init__(setup, figure, axes)

        self.right = self.ax.twinx()
        self.axes_dict["right"] = self.right

        self.right.set_yscale(self.setup.yscale)
        if self.setup.rlabel is not None:
            self.right.set_ylabel(self.setup.rlabel)
        if self.setup.rlim is not None:
            self.right.set_ylim(self.setup.rlim)

        return None

    def postprocess(self) -> None:

        super().postprocess()

        for line in self.right.get_lines():
            line.set_color(next(self.color_cycler)["color"])


class ParasiteAxis(DoubleAxis):

    def __init__(self, setup: SubplotSetup, figure: mplFigure, axes: Axes) -> None:

        super().__init__(setup, figure, axes)

        self.parax = self.ax.twinx()
        self.parax.spines.right.set_position(("axes", 1.13))
        self.axes_dict["parax"] = self.parax

        self.parax.set_yscale(self.setup.pscale)
        if self.setup.plabel is not None:
            self.parax.set_ylabel(self.setup.plabel)
        if self.setup.plim is not None:
            self.parax.set_ylim(self.setup.plim)

        return None

    def postprocess(self) -> None:

        super().postprocess()

        for line in self.parax.get_lines():
            line.set_color(next(self.color_cycler)["color"])

        return None
