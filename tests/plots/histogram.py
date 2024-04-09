import nastro.plots as pp
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":

    n02 = np.random.normal(0, 2, 1000)
    n01 = np.random.normal(0, 1, 1000)
    n11 = np.random.normal(1, 1, 1000)

    # plt.hist(n02, bins=20, histtype="step")
    # plt.show()
    # exit(0)

    setup = pp.PlotSetup(
        title="Some Gaussian distributions", xlabel="x", ylabel=r"$p(x)$"
    )
    with pp.SingleAxis(setup) as fig:

        fig.add_histogram(
            n02,
            bins=20,
            normalize=True,
            cumulative=False,
            label=r"$\mu = 0\ \sigma = 2$",
        )
        fig.add_histogram(
            n01,
            bins=20,
            normalize=True,
            cumulative=False,
            label=r"$\mu = 0\ \sigma = 1$",
            alpha=0.5,
            align="right",
        )
        fig.add_histogram(
            n11,
            bins=20,
            normalize=True,
            cumulative=False,
            label=r"$\mu = 1\ \sigma = 1$",
            alpha=1.0,
            hist_type="step",
        )
