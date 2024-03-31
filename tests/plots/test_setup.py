import nastro.plots as npt


def test_copy() -> None:
    """Test copy method of Plot class. [Robust]"""

    plot = npt.PlotSetup(save=True)
    plot_copy = plot.copy()

    for key in plot.__dict__.keys():
        assert plot.__dict__[key] == plot_copy.__dict__[key]

    return None
