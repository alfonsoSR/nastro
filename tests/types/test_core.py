import nastro.types.core as nt  # type: ignore
import numpy as np


def test_isinstance() -> None:

    # Scalar
    assert nt.is_scalar(1)
    assert nt.is_scalar(1.0)
    assert nt.is_scalar(np.int32(1))
    assert nt.is_scalar(np.int64(1))
    assert nt.is_scalar(np.float64(1.0))
    assert nt.is_scalar(np.float128(1.0))
    assert not nt.is_scalar(np.array([1.0]))
    assert not nt.is_scalar(np.array([1]))
    assert not nt.is_scalar([1.0])

    # Double
    assert not nt.is_double(1)
    assert nt.is_double(1.0)
    assert not nt.is_double(np.int32(1))
    assert not nt.is_double(np.int64(1))
    assert nt.is_double(np.float64(1.0))
    assert not nt.is_double(np.float128(1.0))
    assert not nt.is_double(np.array([1.0]))
    assert not nt.is_double(np.array([1]))
    assert not nt.is_double([1.0])

    # Vector
    assert not nt.is_vector(1)
    assert not nt.is_vector(1.0)
    assert not nt.is_vector(np.int32(1))
    assert not nt.is_vector(np.int64(1))
    assert not nt.is_vector(np.float64(1.0))
    assert not nt.is_vector(np.float128(1.0))
    assert nt.is_vector(np.array([1.0]))
    assert not nt.is_vector(np.array([1]))
    assert not nt.is_vector([1.0])

    # Array
    assert not nt.is_array(1)
    assert not nt.is_array(1.0)
    assert not nt.is_array(np.int32(1))
    assert not nt.is_array(np.int64(1))
    assert not nt.is_array(np.float64(1.0))
    assert not nt.is_array(np.float128(1.0))
    assert nt.is_array(np.array([1.0]))
    assert nt.is_array(np.array([1]))
    assert nt.is_array([1.0])

    return None
