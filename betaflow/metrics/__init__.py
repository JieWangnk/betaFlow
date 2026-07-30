"""Error metrics, looked up by the name used in case YAML files."""

from betaflow.metrics.norms import l2_phase, l2_velocity, relative_error_scalar

METRICS = {
    "L2_velocity": l2_velocity,
    "wss_relative": relative_error_scalar,
    # Pulsatile cases: amplitude reuses the discrete L2 norm; phase is
    # amplitude-weighted; WSS amplitude reuses the scalar relative error.
    "L2_amplitude": l2_velocity,
    "L2_phase": l2_phase,
    "wss_amp_relative": relative_error_scalar,
}
