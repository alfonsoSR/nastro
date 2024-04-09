import nastro.plots as pp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    z = np.cos(x)
    error = np.abs(np.random.normal(0.0, 0.1, x.shape))
    limit = 0.5
    setup = pp.PlotSetup(title="Boundaries demo")

    with pp.Mosaic("ab;cd", setup) as fig:

        # Adds scalar boundary to last line
        with fig.add_subplot(setup=pp.PlotSetup(grid=True)) as a:

            a.add_line(x, y, label="sin", color="orange")
            a.add_boundary(limit, follow=True)
            a.add_line(x, z, label="cos", fmt=".", markersize=1)
            a.add_boundary(limit, follow=True)

        # Adds ArrayLike boundary independent of any line
        with fig.add_subplot() as b:

            b.add_boundary(error, color="green", line=None, label="error")

        # Adds ArrayLike boundary associated with line [Follow & Not Follow]
        with fig.add_subplot() as c:

            c.add_line(x, z, label="cos")
            c.add_boundary(error, line="cos")
            c.add_boundary(error, color="green", line="cos", follow=True)

        # Vertical boundary
        with fig.add_subplot() as d:

            d.add_line(x, y, label="sin")
            d.add_vertical_boundary(2.0, 4.0, color="green")
            d.add_vertical_boundary(1.0, 5.0)
