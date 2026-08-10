"""Error norms. Inputs are plain arrays already non-dimensionalised by u_ref."""

import numpy as np


def l2_velocity(numerical, analytic):
    """Discrete L2 norm of the velocity error at the sample points.

    Both inputs must be non-dimensional (u/u_max), evaluated at the same
    wall-normal stations, so the result is already normalised by u_max:

        E = sqrt( mean( (u_num/u_max - u_exact/u_max)**2 ) )
    """
    num = np.asarray(numerical, dtype=float)
    ana = np.asarray(analytic, dtype=float)
    if num.shape != ana.shape:
        raise ValueError(f"shape mismatch: numerical {num.shape} vs analytic {ana.shape}")
    return float(np.sqrt(np.mean((num - ana) ** 2)))


def l2_phase(numerical, analytic, weights):
    """Amplitude-weighted RMS phase error [radians] at the sample points.

    Phase differences are wrapped to (-pi, pi] before averaging, and weighted
    (typically by the squared analytic amplitude) because phase is
    ill-conditioned where the amplitude vanishes.
    """
    num = np.asarray(numerical, dtype=float)
    ana = np.asarray(analytic, dtype=float)
    w = np.asarray(weights, dtype=float)
    if not (num.shape == ana.shape == w.shape):
        raise ValueError("shape mismatch between phases and weights")
    diff = np.angle(np.exp(1j * (num - ana)))
    return float(np.sqrt(np.sum(w * diff**2) / np.sum(w)))


def relative_error_scalar(numerical, analytic):
    """|numerical - analytic| / |analytic| for a scalar quantity.

    Both values must be in the same units; the result is dimensionless.
    """
    ana = float(analytic)
    if ana == 0.0:
        raise ValueError("analytic reference is zero; relative error undefined")
    return abs(float(numerical) - ana) / abs(ana)


def _edge_from_mask(y, inside):
    """Half-width implied by a boolean 'inside the plug' mask on cell centres.

    Returns the face position between the outermost inside cell and the
    innermost outside cell — the natural discrete boundary. The measurement is
    therefore quantised to the cell size, which is what couples the mesh and
    regularisation axes: a cap defect smaller than dy cannot be seen.
    """
    ay = np.abs(np.asarray(y, dtype=float))
    inside = np.asarray(inside, dtype=bool)
    if not inside.any():
        return 0.0
    y_in = ay[inside].max()
    outside = ay[~inside & (ay > y_in)]
    if outside.size == 0:
        return float(ay.max())
    return float(0.5 * (y_in + outside.min()))


def plug_width_cap_active(y, nu_eff, nu_max, rtol=1e-9):
    """Plug half-width from the CAP-ACTIVE region (nu == nuMax) — primary.

    Mechanism-direct: it measures exactly the region where the regularisation
    is doing the work, using the solver's own viscosity field.
    """
    return _edge_from_mask(y, np.asarray(nu_eff) >= nu_max * (1.0 - rtol))


def plug_width_flatness(y, u, tol=1e-3):
    """Plug half-width from profile FLATNESS (|u - u_centre|/u_max < tol).

    Secondary, and threshold-dependent by construction: it detects where the
    profile is flat to within `tol`, which is not the same question as where
    the viscosity cap is active. The gap between the two is informative.
    """
    u = np.asarray(u, dtype=float)
    ay = np.abs(np.asarray(y, dtype=float))
    u_centre = u[np.argmin(ay)]
    u_max = np.max(np.abs(u))
    return _edge_from_mask(y, np.abs(u - u_centre) / u_max < tol)


def plug_velocity_variation(y, u, y_p):
    """max|u - u_centre| / u_max inside the true plug |y| < y_p.

    The capped plug is not rigid: it creeps at gammadot = G y / nuMax, so this
    residual variation scales as 1/nuMax — a different exponent from the plug
    width, from the same cap.
    """
    u = np.asarray(u, dtype=float)
    ay = np.abs(np.asarray(y, dtype=float))
    inside = ay < y_p
    if not inside.any():
        return 0.0
    u_centre = u[np.argmin(ay)]
    return float(np.max(np.abs(u[inside] - u_centre)) / np.max(np.abs(u)))


def msd_slope_relative(msd_values, times, diffusivity):
    """|fitted MSD slope / (6 D) - 1| for free 3-D Brownian motion.

    The fit is through the origin, because MSD(0) = 0 exactly by construction
    — fitting an intercept would absorb part of the signal being tested.
    The residual error here is STATISTICAL (order 1/sqrt(N)), not a
    discretisation error, so it does not shrink with timestep.
    """
    t = np.asarray(times, dtype=float)
    m = np.asarray(msd_values, dtype=float)
    if t.shape != m.shape:
        raise ValueError("times and msd must have the same shape")
    slope = float(np.sum(t * m) / np.sum(t * t))
    return abs(slope / (6.0 * float(diffusivity)) - 1.0)


def d_eff_slope(var_x, times, window):
    """Fit sigma_x^2 = 2 D_eff t + c over `window` and return D_eff.

    An INTERCEPT is fitted, unlike the free-diffusion MSD: the ballistic
    transient leaves a permanent offset in sigma_x^2 that the asymptotic law
    does not describe, and forcing the fit through the origin would fold that
    offset into the slope.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(var_x, dtype=float)
    sel = (t >= window[0]) & (t <= window[1])
    if sel.sum() < 3:
        raise ValueError("fit window contains fewer than three points")
    slope, _ = np.polyfit(t[sel], v[sel], 1)
    return float(slope / 2.0)


def d_eff_relative(var_x, times, window, d_eff_exact):
    """|fitted D_eff / exact - 1|."""
    return abs(d_eff_slope(var_x, times, window) / float(d_eff_exact) - 1.0)


def short_time_exponent(var_x, times, window, diffusivity):
    """(exponent, prefactor) of the shear-sampling regime.

    Fits log(sigma_x^2 - 2 D t) against log t. The 2 D t is SUBTRACTED because
    it is known exactly and additive: sigma_x^2 = 2 D t + Var(u) t^2 at short
    times, so leaving it in makes the raw log-log slope drift below 2 wherever
    the two terms are comparable, which would look like a failure of the
    ballistic law rather than the presence of a second known term.
    """
    t = np.asarray(times, dtype=float)
    v = np.asarray(var_x, dtype=float) - 2.0 * float(diffusivity) * t
    sel = (t >= window[0]) & (t <= window[1]) & (v > 0.0)
    if sel.sum() < 3:
        raise ValueError("short-time window contains fewer than three usable points")
    slope, intercept = np.polyfit(np.log(t[sel]), np.log(v[sel]), 1)
    return float(slope), float(np.exp(intercept))


def radial_ks(r_over_a):
    """Kolmogorov-Smirnov statistic of sampled r/a against F(r) = (r/a)^2."""
    from scipy.stats import kstest

    from betaflow.analytic.taylor_aris import radial_cdf

    return float(kstest(np.asarray(r_over_a, dtype=float), radial_cdf).statistic)


def _line_fit(values, times, window):
    """(slope, intercept) of a straight-line fit over `window`."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    sel = (t >= window[0]) & (t <= window[1])
    if sel.sum() < 3:
        raise ValueError("fit window contains fewer than three points")
    slope, intercept = np.polyfit(t[sel], v[sel], 1)
    return float(slope), float(intercept)


def variance_intercept_relative(var_x, times, window, intercept_exact):
    """|fitted intercept of sigma_x^2 / exact - 1|.

    The INTERCEPT, not the slope. Once D_eff is a free parameter any axial
    numerical diffusion is invisible in the slope, because it simply enlarges
    the fitted D_eff. The intercept weights the transverse eigenspectrum as
    beta^-8 where D_eff weights it as beta^-6, so fitting both imposes two
    independent constraints on the same operator instead of one.
    """
    _, intercept = _line_fit(var_x, times, window)
    return abs(intercept / float(intercept_exact) - 1.0)


def cumulant_slope_relative(kappa3, times, window, slope_exact):
    """|fitted d(kappa_3)/dt / exact - 1|.

    The third cumulant is EXACTLY blind to axial molecular diffusion: any
    symmetric axial kernel contributes nothing beyond the second cumulant. So
    unlike the variance it needs no 2 D t subtracted before it says anything,
    and that subtraction is precisely where scheme-induced axial diffusion
    hides. Its sign is geometry-dependent, which no other anchor here is.
    """
    slope, _ = _line_fit(kappa3, times, window)
    return abs(slope / float(slope_exact) - 1.0)


def centroid_relative(centroid, times, u_mean, offset_exact=0.0):
    """|max deviation of the centroid from U t + offset| / (U * t_end).

    Pass the DISCRETE mean velocity, not the nominal one. A scheme advances
    the centroid at the mean velocity its own quadrature realises, and the
    difference accumulates as a linear drift: read at a single late time it
    is indistinguishable from an offset error, and it doubles when the run
    doubles in length. Comparing against the discrete value isolates the
    centroid behaviour from the velocity quadrature.

    For a cross-sectionally uniform release the centroid is U t EXACTLY at all
    times, independent of D, Pe, mesh and numerical diffusion, because a
    uniform transverse profile is its own steady state. That makes it a pure
    test of discrete advection consistency: a first-order upwind scheme
    smeared beyond usefulness still returns U t to round-off, while a wrong
    cell-volume weighting or an inconsistent flux reconstruction does not.
    None of that is visible in sigma_x^2, which absorbs it into an apparent
    dispersivity.
    """
    t = np.asarray(times, dtype=float)
    c = np.asarray(centroid, dtype=float)
    scale = abs(float(u_mean) * t[-1])
    return float(np.max(np.abs(c - (float(u_mean) * t + float(offset_exact)))) / scale)
