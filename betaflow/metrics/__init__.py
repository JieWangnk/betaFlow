"""Error metrics, looked up by the name used in case YAML files."""

from betaflow.metrics.norms import (
    centroid_relative,
    cumulant_slope_relative,
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
    variance_intercept_relative,
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
    # Eulerian scalar transport. d_eff_relative is REUSED unchanged from the
    # Lagrangian case -- same oracle, same metric, different runner -- which
    # is the cross-check made concrete at the metric level.
    "variance_intercept_relative": variance_intercept_relative,
    "cumulant_slope_relative": cumulant_slope_relative,
    "centroid_relative": centroid_relative,
    # Molecular-communications channel impulse response. Both REUSE existing
    # functions: the CIR is a dimensionless fraction curve, so its RMSE
    # against the oracle IS the discrete L2 norm, and the peak/tail-mass
    # comparisons are scalar relative errors. The error laws are binomial —
    # see betaflow/metrics/mc_error.py (binomial_sigma, binomial_rmse_floor,
    # binomial_integral_sigma_bound).
    "cir_rmse": l2_velocity,
    "cir_peak_relative": relative_error_scalar,
    "cir_tail_mass_relative": relative_error_scalar,
}
