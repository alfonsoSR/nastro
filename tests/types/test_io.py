from nastro.types import CartesianState, KeplerianState
from pathlib import Path
import numpy as np

DATADIR = Path(__file__).parents[1] / "data"


def test_save_keplerian():

    tmp = DATADIR / "tmp.npy"
    try:
        kstate = KeplerianState.load(DATADIR / "kstate.npy")
        kstate.save(tmp)
        loaded = KeplerianState.load(tmp)

        assert kstate == loaded
    finally:
        tmp.unlink()


def test_load_keplerian():

    kstate = KeplerianState.load(DATADIR / "kstate.npy")
    data = np.load(DATADIR / "kstate.npy")
    reference = KeplerianState(*data)

    assert isinstance(kstate, KeplerianState)
    assert kstate == reference


def test_load_cartesian():

    cstate = CartesianState.load(DATADIR / "cstate.npy")
    data = np.load(DATADIR / "cstate.npy")
    reference = CartesianState(*data)

    assert isinstance(cstate, CartesianState)
    assert cstate == reference


def test_save_cartesian():

    tmp = DATADIR / "tmp.npy"
    try:
        cstate = CartesianState.load(DATADIR / "cstate.npy")
        cstate.save(tmp)
        loaded = CartesianState.load(tmp)

        assert cstate == loaded
    finally:
        tmp.unlink()
