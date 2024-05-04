from nastro import types as nt, constants as nc
import numpy as np
import pytest


@pytest.mark.parametrize(
    ["_type", "_input", "_properties"],
    [
        (
            nt.CartesianState,
            np.random.normal(size=(6, 1)),
            ["x", "y", "z", "dx", "dy", "dz"],
        ),
        (
            nt.CartesianState,
            np.random.normal(size=(6, 10)),
            ["x", "y", "z", "dx", "dy", "dz"],
        ),
        (
            nt.CartesianStateDerivative,
            np.random.normal(size=(6, 1)),
            ["dx", "dy", "dz", "ddx", "ddy", "ddz"],
        ),
        (
            nt.CartesianStateDerivative,
            np.random.normal(size=(6, 10)),
            ["dx", "dy", "dz", "ddx", "ddy", "ddz"],
        ),
        (
            nt.KeplerianState,
            np.random.normal(size=(6, 1)),
            ["a", "e", "i", "raan", "aop", "ta"],
        ),
        (
            nt.KeplerianState,
            np.random.normal(size=(6, 10)),
            ["a", "e", "i", "raan", "aop", "ta"],
        ),
        (
            nt.KeplerianStateDerivative,
            np.random.normal(size=(6, 1)),
            ["da", "de", "di", "draan", "daop", "dta"],
        ),
        (
            nt.KeplerianStateDerivative,
            np.random.normal(size=(6, 10)),
            ["da", "de", "di", "draan", "daop", "dta"],
        ),
        (
            nt.CartesianPosition,
            np.random.normal(size=(3, 1)),
            ["x", "y", "z"],
        ),
        (
            nt.CartesianPosition,
            np.random.normal(size=(3, 10)),
            ["x", "y", "z"],
        ),
        (
            nt.CartesianVelocity,
            np.random.normal(size=(3, 1)),
            ["dx", "dy", "dz"],
        ),
        (
            nt.CartesianVelocity,
            np.random.normal(size=(3, 10)),
            ["dx", "dy", "dz"],
        ),
    ],
    ids=[
        "cstate-scalar",
        "cstate-vector",
        "dcstate-scalar",
        "dcstate-vector",
        "kstate-scalar",
        "kstate-vector",
        "dkstate-scalar",
        "dkstate-vector",
        "cpos-scalar",
        "cpos-vector",
        "cvel-scalar",
        "cvel-vector",
    ],
)
def test_basic_properties(_type, _input, _properties) -> None:

    state = _type(*_input)
    wrapped_input = []
    for val, prop in zip(_input, _properties):
        if prop in _type.angles:
            wrapped_input.append(
                getattr(nt.GenericState, f"_GenericState__wrap_angle")(
                    val, _type.angles[prop]
                )
            )
        else:
            wrapped_input.append(val)

    for idx, prop in enumerate(_properties):
        assert np.allclose(getattr(state, prop), wrapped_input[idx])
    assert state.scalar if _input.shape[1] == 1 else not state.scalar
    assert np.allclose(state.asarray, np.array(wrapped_input))

    for idx, prop in enumerate(_properties):
        if prop in _type.angles:
            assert np.allclose(
                getattr(state, f"{prop}_deg"), np.rad2deg(wrapped_input[idx])
            )

    return None


@pytest.mark.parametrize(
    ["_type", "_input", "_properties"],
    [
        (
            nt.CartesianState,
            np.random.normal(size=(6, 1)),
            ["r_vec", "v_vec", "r_mag", "v_mag"],
        ),
        (
            nt.CartesianState,
            np.random.normal(size=(6, 10)),
            ["r_vec", "v_vec", "r_mag", "v_mag"],
        ),
        (
            nt.CartesianStateDerivative,
            np.random.normal(size=(6, 1)),
            ["v_vec", "a_vec", "v_mag", "a_mag"],
        ),
        (
            nt.CartesianStateDerivative,
            np.random.normal(size=(6, 10)),
            ["v_vec", "a_vec", "v_mag", "a_mag"],
        ),
    ],
    ids=["cstate-scalar", "cstate-vector", "dcstate-scalar", "dcstate-vector"],
)
def test_vectors_and_magnitudes(_type, _input, _properties) -> None:

    state = _type(*_input)
    vec1, vec2, mag1, mag2 = _properties
    assert np.allclose(getattr(state, vec1), _input[:3])
    assert np.allclose(getattr(state, vec2), _input[3:])
    assert np.allclose(getattr(state, mag1), np.linalg.norm(_input[:3], axis=0))
    assert np.allclose(getattr(state, mag2), np.linalg.norm(_input[3:], axis=0))

    return None


@pytest.mark.parametrize(
    ["_dtype", "_type", "_input"],
    [
        (
            nt.CartesianStateDerivative,
            nt.CartesianState,
            np.random.normal(size=(6, 1)),
        ),
        (
            nt.CartesianStateDerivative,
            nt.CartesianState,
            np.random.normal(size=(6, 10)),
        ),
        (
            nt.KeplerianStateDerivative,
            nt.KeplerianState,
            np.random.normal(size=(6, 1)),
        ),
        (
            nt.KeplerianStateDerivative,
            nt.KeplerianState,
            np.random.normal(size=(6, 10)),
        ),
    ],
    ids=["cstate-scalar", "cstate-vector", "kstate-scalar", "kstate-vector"],
)
def test_times_dt(_dtype, _type, _input) -> None:
    """s = ds * dt"""

    ds = _dtype(*_input)
    s = ds.times_dt(10.0)
    wrapped_input = []
    for idx, val in enumerate(_input):
        key = list(_type.properties.keys())[idx]
        if key in _type.angles:
            wrapped_input.append(
                getattr(nt.GenericState, f"_GenericState__wrap_angle")(
                    val * 10.0, _type.angles[key]
                )
            )
        else:
            wrapped_input.append(val * 10.0)

    assert isinstance(s, _type)
    assert np.allclose(s.asarray, np.array(wrapped_input))


@pytest.mark.parametrize(
    ["_input"],
    [
        (np.random.normal(size=(6, 1)),),
        (np.random.normal(size=(6, 10)),),
    ],
    ids=["scalar", "vector"],
)
def test_add_acceleration(_input) -> None:

    ds = nt.CartesianStateDerivative(*_input)

    if ds.scalar:
        new = ds.copy()
        new.add_acceleration(ds)
    else:
        with pytest.raises(NotImplementedError):
            ds.add_acceleration(ds)
        return None

    assert isinstance(new, nt.CartesianStateDerivative)
    assert new.scalar
    assert np.allclose(new.v_vec, _input[:3])
    assert np.allclose(new.a_vec, _input[3:] * 2.0)


@pytest.mark.skip
@pytest.mark.parametrize(
    ["property"],
    [("E"), ("M"), ("T")],
)
def test_KeplerianState_properties(property) -> None:

    scalar_input = np.random.normal(size=(6, 1))
    scalar_state = nt.KeplerianState(*scalar_input, wrap=False)
    vector_input = np.random.normal(size=(6, 10))
    vector_state = nt.KeplerianState(*vector_input, wrap=False)

    return None
