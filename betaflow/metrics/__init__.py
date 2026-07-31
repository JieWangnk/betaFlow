"""Error metrics, looked up by the name used in case YAML files."""

from betaflow.metrics.norms import (
    l2_phase,
    l2_velocity,
    d_eff_relative,
    d_eff_slope,
    msd_slope_relative,
    radial_ks,
    short_time_exponent,
    plug_velocity_variation,
    plug_width_cap_active,
    plug_width_flatness,
    relative_error_scalar,
)

METRICS = {
    "L2_velocity": l2_velocity,
    "wss_relative": relative_error_scalar,
    # Pulsatile cases: amplitude reuses the discrete L2 norm; phase is
    # amplitude-weighted; WSS amplitude reuses the scalar relative error.
    "L2_amplitude": l2_velocity,
    "L2_phase": l2_phase,
    "wss_amp_relative": relative_error_scalar,
    # Yield-stress cases: two independent plug-width definitions plus the
    # residual creep the regularisation cap leaves behind.
    "plug_width_cap_active": plug_width_cap_active,
    "plug_width_flatness": plug_width_flatness,
    "plug_velocity_variation": plug_velocity_variation,
    # Particle transport: the error is statistical, not discretisation.
    "msd_slope_relative": msd_slope_relative,
    # Taylor-Aris: a fitted variance, a different power of t, and a
    # DISTRIBUTION-valued check (scalar-valued output, so the {name, tol}
    # case schema needed no change to express it).
    "d_eff_relative": d_eff_relative,
    "d_eff_slope": d_eff_slope,
    "short_time_exponent": short_time_exponent,
    "radial_ks": radial_ks,
}
