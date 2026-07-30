"""Error metrics, looked up by the name used in case YAML files."""

from betaflow.metrics.norms import l2_velocity, relative_error_scalar

METRICS = {
    "L2_velocity": l2_velocity,
    "wss_relative": relative_error_scalar,
}
