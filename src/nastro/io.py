from .types import JulianDay, CartesianState, Double, Vector


def tudat_state_history(
    epochs: JulianDay, states: CartesianState
) -> dict[Double, Vector]:
    """Generate tudat cartesian state history from epochs and states."""

    return {epoch.jd: state.asarray for epoch, state in zip(epochs, states)}
