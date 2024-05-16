from nastro import types as nt, constants as nc
import numpy as np
import pytest
from pathlib import Path


def test_generic_properties() -> None:

    # Scalar input
    scalar_input = np.random.normal(size=(6,))
    scalar_state = nt.GenericState(*scalar_input)
    with pytest.raises(AttributeError):
        scalar_state.__dict__
    assert np.allclose(scalar_state.q1, scalar_input[0])
    assert np.allclose(scalar_state.q2, scalar_input[1])
    assert np.allclose(scalar_state.q3, scalar_input[2])
    assert np.allclose(scalar_state.q4, scalar_input[3])
    assert np.allclose(scalar_state.q5, scalar_input[4])
    assert np.allclose(scalar_state.q6, scalar_input[5])
    assert scalar_state.scalar
    assert np.allclose(scalar_state.asarray, scalar_input[:, None])

    # Vector input
    vector_input = np.random.normal(size=(6, 10))
    vector_state = nt.GenericState(*vector_input)
    assert np.allclose(vector_state.q1, vector_input[0])
    assert np.allclose(vector_state.q2, vector_input[1])
    assert np.allclose(vector_state.q3, vector_input[2])
    assert np.allclose(vector_state.q4, vector_input[3])
    assert np.allclose(vector_state.q5, vector_input[4])
    assert np.allclose(vector_state.q6, vector_input[5])
    assert not vector_state.scalar
    assert np.allclose(vector_state.asarray, vector_input)

    return None


def test_getitem_and_iter() -> None:

    array_reference = np.random.normal(size=(6, 10))
    state_reference = nt.GenericState(*array_reference)

    for idx in range(10):
        assert np.allclose(
            state_reference[idx].asarray.flatten(), array_reference[:, idx]
        )

    for si, refi in zip(state_reference, array_reference.T):
        assert np.allclose(si.asarray.flatten(), refi)


def test_addition() -> None:

    array_input = np.random.normal(size=(6, 10))
    reference = nt.GenericState(*array_input)
    integral_input = 4
    double_input = 4.0
    state_input = nt.GenericState(*(4.0 * np.ones((6, 10))))

    array_expected = array_input + 4.0

    assert np.allclose(array_expected, (reference + integral_input).asarray)
    assert np.allclose(array_expected, (reference + double_input).asarray)
    assert np.allclose(array_expected, (reference + state_input).asarray)


def test_subtraction() -> None:

    array_input = np.random.normal(size=(6, 10))
    reference = nt.GenericState(*array_input)
    integral_input = 4
    double_input = 4.0
    state_input = nt.GenericState(*(4.0 * np.ones((6, 10))))

    array_expected = array_input - 4.0

    assert np.allclose(array_expected, (reference - integral_input).asarray)
    assert np.allclose(array_expected, (reference - double_input).asarray)
    assert np.allclose(array_expected, (reference - state_input).asarray)


def test_multiplication() -> None:

    array_input = np.random.normal(size=(6, 10))
    reference = nt.GenericState(*array_input)
    integral_input = 4
    double_input = 4.0
    state_input = nt.GenericState(*(4.0 * np.ones((6, 10))))

    array_expected = array_input * 4.0

    assert np.allclose(array_expected, (reference * integral_input).asarray)
    assert np.allclose(array_expected, (reference * double_input).asarray)
    assert np.allclose(array_expected, (reference * state_input).asarray)


def test_division() -> None:

    array_input = np.random.normal(size=(6, 10))
    reference = nt.GenericState(*array_input)
    integral_input = 4
    double_input = 4.0
    state_input = nt.GenericState(*(4.0 * np.ones((6, 10))))

    array_expected = array_input / 4.0

    assert np.allclose(array_expected, (reference / integral_input).asarray)
    assert np.allclose(array_expected, (reference / double_input).asarray)
    assert np.allclose(array_expected, (reference / state_input).asarray)


def test_equal() -> None:

    array_input = np.random.normal(size=(6, 10))
    reference = nt.GenericState(*array_input)
    state_input = nt.GenericState(*array_input)

    assert reference == state_input
    assert reference == state_input + 1e-16
    with pytest.raises(AssertionError):
        assert reference == state_input + 1e-15

    return None


def test_append() -> None:

    array_reference = np.random.normal(size=(6,))
    state_reference = nt.GenericState(*array_reference)

    scalar_input = 3.0
    vector_input = 3.0 * np.ones((6,))
    state_input = nt.GenericState(*vector_input)

    array_expected = np.array([array_reference.T.tolist(), [3.0] * 6]).T
    state_expected = nt.GenericState(*array_expected)

    scalar_out = state_reference.copy()
    scalar_out.append(scalar_input)
    vector_out = state_reference.copy()
    vector_out.append(vector_input)
    state_out = state_reference.copy()
    state_out.append(state_input)

    assert state_expected == scalar_out
    assert state_expected == vector_out
    assert state_expected == state_out


@pytest.mark.parametrize(
    ["path", "rel"],
    [
        (Path(__file__).parent / "tmp.npy", False),
        ("tmp.npy", True),
        (Path(__file__).parent / "tmp", False),
        ("tmp", True),
    ],
    ids=["abs_ext", "rel_ext", "abs_noext", "rel_noext"],
)
def test_io(path: Path, rel: bool) -> None:

    s = nt.GenericState(*np.random.normal(size=(6, 10)))
    actual_path = Path(__file__).parent / "tmp.npy"

    # Absolute path with extension
    outpath = s.save(path, relative=rel)
    assert actual_path.is_file()
    np_loaded = np.load(actual_path)
    s_loaded = nt.GenericState[nt.Vector].load(outpath)
    assert s == s_loaded and np.allclose(s.asarray, np_loaded)
    actual_path.unlink()


def test_copy() -> None:

    s = nt.GenericState(*np.random.normal(size=(6, 10)))
    s_copy = s.copy()
    assert s == s_copy
    assert np.allclose(s.asarray, s_copy.asarray)

    return None


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

        out = getattr(nt.GenericState, "_GenericState__wrap_angle")(i, limits)
        assert np.allclose(out, o)
