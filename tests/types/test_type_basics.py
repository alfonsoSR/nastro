from nastro.types import CartesianState
import nastro.constants as nc
import numpy as np

# TODO: Add more tests


def test_creation():
    """Test creation of CartesianState from single and series of states"""

    _ = CartesianState(1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
