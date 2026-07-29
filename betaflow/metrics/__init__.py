"""Error metrics, looked up by the name used in case YAML files."""

from betaflow.metrics.norms import l2_velocity

METRICS = {
    "L2_velocity": l2_velocity,
}
