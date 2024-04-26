import nastro.plots as ng
import numpy as np

if __name__ == "__main__":

    x = np.arange(5)
    y = np.random.normal(size=len(x))

    setup = ng.PlotSetup(xticks=x[1:])

    with ng.Mosaic("ab") as figure:

        with figure.add_subplot(setup=setup) as plot:
            plot.add_line(x, y, fmt="o-")
