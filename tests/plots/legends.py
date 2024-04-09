import nastro.plots as pp
import numpy as np
from matplotlib import colormaps

if __name__ == "__main__":

    # x = np.linspace(0, 10, 100)
    # y = np.sin(x)
    # z = np.cos(x)

    data = np.random.normal(0, 2, (20, 20))

    # setup = pp.PlotSetup(
    #     title="The title of the figure",
    #     axtitle="The title of the axes",
    #     rlabel="This one's gonna give problems",
    #     show=False,
    # )

    # with pp.ParasiteAxis(setup) as fig:

    #     fig.add_line(x, y, label="sin(x)")
    #     fig.add_line(x, z, label="cos(x)", axis="right")
    #     fig.add_line(x, y + z, label="sin(x) + cos(x)", axis="parasite")

    setup = pp.PlotSetup(
        figsize=(5.0, 4.0),
        title="Correlation matrix",
        cbar_label="Description of the colorbar",
        cbar_shrink=0.8,
    )

    with pp.SingleAxis(setup) as fig:

        fig.add_image(data)
