import numpy as np
from nastro import types as nt
import pytest


def test_keplerian2cartesian():
    """Test conversion from Keplerian to Cartesian state. [Robust]"""

    mu_sun = 1.3271276489987138e20

    original_kstate = nt.KeplerianState(
        1.082073518745249e11,
        6.737204890591715e-03,
        3.394387859410573e00,
        7.661212333458995e01,
        5.520309419058912e01,
        3.559064425112629e02,
        deg=True,
    )

    expected_cstate = nt.CartesianState(
        -6.561639868572587e10,
        8.498200549242477e10,
        4.953209922912188e09,
        -2.784002901222631e04,
        -2.159352048214584e04,
        1.309840720051276e03,
    )

    converted_cstate = original_kstate.to_cartesian(mu_sun)
    converted_kstate = converted_cstate.to_keplerian(mu_sun)

    # Relative errors
    TOL = 2.2e-15

    assert np.isclose(converted_cstate.x, expected_cstate.x, rtol=TOL)
    assert np.isclose(converted_cstate.y, expected_cstate.y, rtol=TOL)
    assert np.isclose(converted_cstate.z, expected_cstate.z, rtol=TOL)
    assert np.isclose(converted_cstate.dx, expected_cstate.dx, rtol=TOL)
    assert np.isclose(converted_cstate.dy, expected_cstate.dy, rtol=TOL)
    assert np.isclose(converted_cstate.dz, expected_cstate.dz, rtol=TOL)

    assert np.isclose(converted_kstate.a, original_kstate.a, rtol=TOL)
    assert np.isclose(converted_kstate.e, original_kstate.e, rtol=TOL)
    assert np.isclose(converted_kstate.i, original_kstate.i, rtol=TOL)
    assert np.isclose(converted_kstate.raan, original_kstate.raan, rtol=TOL)
    assert np.isclose(converted_kstate.aop, original_kstate.aop, rtol=TOL)
    assert np.isclose(converted_kstate.ta, original_kstate.ta, rtol=TOL)


@pytest.mark.skip
def test_time_to_mean_anomaly() -> None:
    """Test conversion from time to mean anomaly"""

    raise NotImplementedError


@pytest.mark.skip
def test_true_to_eccentric_anomaly() -> None:
    """Test conversion from true to eccentric anomaly"""

    raise NotImplementedError


@pytest.mark.skip
def test_eccentric_to_true_anomaly() -> None:
    """Test conversion from eccentric to true anomaly"""

    raise NotImplementedError


@pytest.mark.skip
def test_eccentric_to_mean_anomaly() -> None:
    """Test conversion from eccentric to mean anomaly"""

    raise NotImplementedError
