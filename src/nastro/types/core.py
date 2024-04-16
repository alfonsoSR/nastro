import numpy as np
from numpy.typing import NDArray
from typing import Sequence, TypeVar

Double = np.floating | float
Vector = NDArray[np.floating]
ArrayLike = np.ndarray | Sequence
T = TypeVar("T", Double, Vector)
