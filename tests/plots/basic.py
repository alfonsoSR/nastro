from nastro import graphics as ng
import numpy as np


if __name__ == "__main__":

    setup = ng.PlotSetup(
        canvas_layout="constrained",
        canvas_size=(7, 7),
    )

    left_setup = ng.PlotSetup(
        xlabel="X [AU]",
        ylabel="Y [AU]",
    )

    setup_3d = ng.PlotSetup(
        xlabel="X [AU]",
        ylabel="Y [AU]",
        zlabel="Z [AU]",
    )

    data = np.random.uniform(0.9, 0.999, (3, 100))

    # with ng.SingleAxis(setup) as single:
    #     pass

    with ng.Mosaic("ab;cd", setup) as mosaic:

        with mosaic.subplot(left_setup, ng.SingleAxis) as single:
            single.line(data[0], data[1], fmt="o")

        with mosaic.subplot(setup_3d, generator=ng.Plot3D) as sub:
            sub.line(*data, fmt="o")

        with mosaic.subplot(left_setup, ng.SingleAxis) as single:
            single.line(data[0], data[1], fmt="o")

        with mosaic.subplot(setup_3d, generator=ng.Plot3D) as sub:
            sub.line(*data, fmt="o")
