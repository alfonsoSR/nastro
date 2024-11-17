import numpy as np
import numpy.typing as npt
from typing import Sequence, TypeVar, Any, TypeGuard

Scalar = int | float | np.floating | np.integer
Double = float | np.float64
Vector = npt.NDArray[np.float64]
Array = npt.NDArray[Any] | Sequence[int | float]
T = TypeVar("T", Double, Vector)


def is_scalar(x: Any) -> TypeGuard[Scalar]:
    return isinstance(x, int) or isinstance(x, float) or isinstance(x, np.number)


def is_double(x: Any) -> TypeGuard[Double]:
    return isinstance(x, float)


def is_vector(x: Any) -> TypeGuard[Vector]:
    return isinstance(x, np.ndarray) and is_double(x[0])


def is_array(x: Any) -> TypeGuard[Array]:
    return (isinstance(x, np.ndarray) or isinstance(x, Sequence)) and is_scalar(x[0])
