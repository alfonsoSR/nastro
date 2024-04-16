from nastro.types import CartesianState
from nastro.types.newstate import GenericState
import nastro.constants as nc
import numpy as np
import pytest

ATOL = 2e-15
RTOL = 0.0


def allclose(a, b):
    return np.allclose(a, b, atol=ATOL, rtol=RTOL)


def test_creation():
    """Test creation of CartesianState from single and series of states"""

    _ = CartesianState(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)


@pytest.mark.parametrize(
    "limits, expected_positive, expected_negative",
    [
        (
            (0.0, nc.twopi),
            np.deg2rad([0.0, 60.0, 150.0, 240.0, 330.0, 0.0]),
            np.deg2rad([0.0, 300.0, 210.0, 120.0, 30.0, 0.0]),
        ),
        (
            (0.0, nc.pi),
            np.deg2rad([0.0, 60.0, 150.0, 60.0, 150.0, 0.0]),
            np.deg2rad([0.0, 120.0, 30.0, 120.0, 30.0, 0.0]),
        ),
        (
            (-nc.pi, nc.pi),
            np.deg2rad([0.0, 60.0, 150.0, -120.0, -30.0, 0.0]),
            np.deg2rad([0.0, -60.0, -150.0, 120.0, 30.0, 0.0]),
        ),
        (
            (-nc.halfpi, nc.halfpi),
            np.deg2rad([0.0, 60.0, 30.0, -60.0, -30.0, 0.0]),
            np.deg2rad([0.0, -60.0, -30.0, 60.0, 30.0, 0.0]),
        ),
    ],
    ids=("0, 360", "0, 180", "-180, 180", "-90, 90"),
)
def test_angle_wrapping(limits, expected_positive, expected_negative) -> None:
    """Test angle wrapping [Robust]

    Includes positive and negative angles, as well as angles that are out of the
    (-360, 360) range.
    """

    # Common input
    positive_input = np.deg2rad([0.0, 60.0, 150.0, 240.0, 330.0, 360.0])
    positive_shifted_input = 2.0 * nc.twopi + positive_input
    negative_input = np.deg2rad([0.0, -60.0, -150.0, -240.0, -330.0, -360.0])
    negative_shifted_input = -2.0 * nc.twopi + negative_input

    # Constrain angles
    input = [
        positive_input,
        positive_shifted_input,
        negative_input,
        negative_shifted_input,
    ]
    output = [
        expected_positive,
        expected_positive,
        expected_negative,
        expected_negative,
    ]

    for idx, (i, o) in enumerate(zip(input, output)):
        try:
            assert allclose(GenericState.wrap_angle(i, limits[0], limits[1]), o)
        except AssertionError:
            raise AssertionError(
                f"{idx}\n"
                f"{np.rad2deg(GenericState.wrap_angle(i, limits[0], limits[1]))}\n"
                f"{np.rad2deg(o)}"
            )
