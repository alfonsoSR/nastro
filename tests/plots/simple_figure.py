import nastro.plots as pp
import numpy as np

if __name__ == "__main__":

    # Generate data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    z = np.cos(x)
    error = np.abs(np.random.normal(0.0, 0.1, x.shape))
    xbar = np.arange(8)
    ybar = np.random.normal(5, 2, 8)
    ticks = ["a", "b", "c", "d", "e", "f", "g", "h"]
    gen = np.random.default_rng(1234797)

    # Create figure
    figure_setup = pp.PlotSetup(
        figsize=(12, 7),
        title="A simple figure with different types of subplots",
        save=True,
        dir=".",
        name="plots.png",
        show=False,
    )
    a_setup = pp.PlotSetup(ylabel="left", rlabel="right", plabel="parasite")
    c_setup = pp.PlotSetup(grid=False)
    d_setup = pp.PlotSetup(ylabel="sin(x)", rlabel="cos(x)")

    with pp.Mosaic("ab;cd;ef", figure_setup) as fig:

        with fig.add_subplot(pp.ParasiteAxis, a_setup) as a:

            a.add_line(x, x, fmt=".-", markersize=2, label="x")
            a.add_line(x, x**2, fmt="--", axis="right", label="x^2")
            a.add_line(x, x**3, axis="parasite", label="x^3")

        with fig.add_subplot(pp.SingleAxis) as b:

            b.add_line(x, y, label="sin")
            b.add_line(x, z, label="cos")
            b.add_boundary(0.5, line="sin", follow=True)
            b.add_boundary(0.25, line="sin", follow=True, alpha=0.3)

        with fig.add_subplot(pp.SingleAxis, c_setup) as c:

            c.add_barplot(xbar, ybar, ticks=ticks)

        with fig.add_subplot(pp.DoubleAxis, d_setup) as d:

            d.add_errorbar(x, y, np.abs(gen.normal(0.0, 0.1, x.shape)))
            d.add_errorbar(x, z, np.abs(gen.normal(0.0, 0.1, x.shape)), axis="right")
            d.add_line(x, 0.8 * y)

        with fig.add_subplot(pp.DoubleAxis) as e:

            e.add_step(x, y, fmt=".-")
            e.add_step(x, z, axis="right")

        with fig.add_subplot(pp.SingleAxis) as f:

            f.add_horizontal_barplot(xbar[:4], ybar[:4], ticks=ticks[:4])
