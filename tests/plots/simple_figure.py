import nastro.graphics as ng
import numpy as np

if __name__ == "__main__":

    # Generate data
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.sin(x)
    z = np.cos(x)
    error = np.abs(np.random.normal(0.0, 0.1, x.shape))
    xbar = np.arange(8)
    ybar = np.random.normal(5, 2, 8)
    # ticks = np.arange(8)
    # ticks = np.array(["a", "b", "c", "d", "e", "f", "g", "h"])
    gen = np.random.default_rng(1234797)

    # Create figure
    figure_setup = ng.PlotSetup(
        figsize=(12, 7),
        title="A simple figure with different types of subplots",
        save=False,
        dir=".",
        name="plots.png",
        show=True,
    )
    a_setup = ng.PlotSetup(ylabel="left", rlabel="right", plabel="parasite")
    c_setup = ng.PlotSetup(grid=False)
    d_setup = ng.PlotSetup(ylabel="sin(x)", rlabel="cos(x)")

    with ng.Mosaic("ab;cd;ef", figure_setup) as fig:

        with fig.subplot(a_setup, ng.ParasiteAxis) as a:

            a.line(x, x, fmt=".-", markersize=2, label="x")
            a.line(x, x**2, fmt="--", axis="right", label="x^2")
            a.line(x, x**3, axis="parasite", label="x^3")

        with fig.subplot(generator=ng.SingleAxis) as b:

            b.line(x, y, label="sin")
            b.line(x, z, label="cos")
            b.boundary(0.5, reference="sin")
            b.boundary(0.25, reference="sin", alpha=0.3)

        with fig.subplot(c_setup, ng.SingleAxis) as c:

            c.bar(xbar, ybar)

        with fig.subplot(d_setup, ng.DoubleAxis) as d:

            d.errorbar(x, y, np.abs(gen.normal(0.0, 0.1, x.shape)))
            d.errorbar(x, z, np.abs(gen.normal(0.0, 0.1, x.shape)), axis="right")
            d.line(x, 0.8 * y)

        with fig.subplot(generator=ng.DoubleAxis) as e:

            e.step(x, y, fmt=".-")
            e.step(x, z, axis="right")

        with fig.subplot() as f:

            f.barh(xbar[:4], ybar[:4])
