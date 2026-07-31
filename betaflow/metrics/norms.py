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
