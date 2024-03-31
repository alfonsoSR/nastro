import nastro.plots.core as npt
import nastro.plots.subplots as nps
import numpy as np

if __name__ == "__main__":

    figure_setup = npt.FigureSetup(
        title="The title of the figure",
        grid="1;2",
        show=True,
        layout="tight",
    )

    subplot_setup = npt.SubplotSetup(
        xlabel="The x-axis label",
        ylabel="The y-axis label",
    )

    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    z = np.cos(x)

    with npt.Figure(figure_setup) as fig:

        with fig.add_subplot(nps.ParasiteAxis, subplot_setup) as subplot:

            subplot.add_line(x, y, label="sin")
            subplot.add_boundary(0.4, follow_reference=True, axis="left", alpha=1)
            subplot.add_line(x, z, label="cos", axis="right")
            subplot.add_boundary(0.4, follow_reference=True, axis="right", alpha=1)

        with fig.add_subplot(nps.SingleAxis, subplot_setup) as subplot:

            subplot.add_line(x, x, label="line")
            subplot.add_boundary(0.4, follow_reference=True)
