"""Analytic reference for steady Casson flow in a plane channel.

Geometry: plane channel, half-height h, no-slip walls at y = +/- h, driven by
a constant streamwise pressure gradient G = -dp/dx > 0. Force balance alone
gives the shear-stress distribution, INDEPENDENT of rheology:

    tau(y) = G * y          =>      tau_w = G * h

Casson constitutive law (sqrt form):

    sqrt(tau) = sqrt(tau_y) + sqrt(mu_c * gammadot)      for tau > tau_y
    gammadot  = 0                                        for tau <= tau_y

so the shear rate outside the plug is

    -du/dy = (sqrt(G y) - sqrt(tau_y))^2 / mu_c

which vanishes QUADRATICALLY at the plug edge — the reason the regularised
plug-width error scales as sqrt(nu_c/nuMax) rather than nu_c/nuMax (see
`cap_active_half_width`).

    plug half-width   y_p = tau_y / G                     (exact)
    for y_p <= |y| <= h:
        u(y) = (1/mu_c) [ G (h^2 - y^2)/2
                          - (4/3) sqrt(G tau_y) (h^1.5 - |y|^1.5)
                          + tau_y (h - |y|) ]
    for |y| < y_p:  u = u(y_p)   (rigid plug)

NON-DIMENSIONAL YIELD PARAMETER:

    xi = tau_y / (G h) = y_p / h

xi is the only shape parameter: with U0 = G h^2 / mu_c and Y = y/h,

    u/U0 = f(Y; xi) = (1 - Y^2)/2 - (4/3) sqrt(xi) (1 - |Y|^1.5) + xi (1 - |Y|)

PHYSIOLOGICAL NOTE: blood in a large artery has xi ~ 0.001-0.005, so the plug
occupies well under 1% of the lumen — which is precisely why Casson is
indistinguishable from Newtonian in the aorta at peak systole. This case uses
xi = 0.2 to make the plug measurable at all; it is a numerical-verification
setting, not a physiological one.

UNITS: every function here works in any unit system but requires
CONSISTENCY. Pass
dynamic (tau_y [Pa], mu_c [Pa s], G [Pa/m]) or kinematic (tau0 = tau_y/rho
[m^2/s^2], nu_c = mu_c/rho [m^2/s], G/rho [m/s^2]) throughout. Incompressible
OpenFOAM is kinematic — its Casson `tau0` is tau_y/rho and its `m` is nu_c —
so betaflow stays kinematic end to end.
"""

import numpy as np


def plug_half_width(tau_y, pressure_gradient):
    """y_p = tau_y / G — exact, and independent of the consistency mu_c."""
    return tau_y / pressure_gradient


def shear_rate(y, pressure_gradient, tau_y, mu_c):
    """-du/dy = (sqrt(G|y|) - sqrt(tau_y))^2 / mu_c outside the plug, 0 inside."""
    y = np.abs(np.asarray(y, dtype=float))
    excess = np.sqrt(pressure_gradient * y) - np.sqrt(tau_y)
    return np.where(excess > 0.0, excess**2 / mu_c, 0.0)


def _f(Y, xi):
    """Non-dimensional profile u/(G h^2/mu_c) at Y = y/h."""
    Y = np.abs(np.asarray(Y, dtype=float))
    Y = np.minimum(Y, 1.0)
    Yc = np.maximum(Y, xi)  # inside the plug the profile is frozen at Y = xi
    return (
        (1.0 - Yc**2) / 2.0
        - (4.0 / 3.0) * np.sqrt(xi) * (1.0 - Yc**1.5)
        + xi * (1.0 - Yc)
    )


def velocity(y, pressure_gradient, tau_y, mu_c, h):
    """u(y) in the same unit system as the arguments."""
    xi = tau_y / (pressure_gradient * h)
    return pressure_gradient * h**2 / mu_c * _f(np.asarray(y) / h, xi)


def velocity_profile(y_over_h, xi):
    """Non-dimensional u/u_max at Y = y/h — the framework analytic reference signature."""
    return _f(y_over_h, xi) / _f(0.0, xi)


def u_max_over_U0(xi):
    """Plug (centreline) velocity in units of U0 = G h^2 / mu_c."""
    return float(_f(0.0, xi))


def u_mean_over_U0(xi):
    """Bulk velocity in units of U0 = G h^2 / mu_c (closed form).

    mean over Y in [0, 1] of f(Y; xi) = plug block + sheared integral.
    """
    s = np.sqrt(xi)
    plug = xi * _f(0.0, xi)
    sheared = (
        (1.0 / 3.0 - xi / 2.0 + xi**3 / 6.0)
        - (4.0 / 3.0) * s * (3.0 / 5.0 - xi + (2.0 / 5.0) * xi**2.5)
        + (xi / 2.0 - xi**2 + xi**3 / 2.0)
    )
    return float(plug + sheared)


def pressure_gradient_for_bulk(u_mean, h, mu_c, xi):
    """G such that the Casson solution with this xi has bulk velocity u_mean."""
    return u_mean * mu_c / (h**2 * u_mean_over_U0(xi))


def tau_wall(pressure_gradient, h):
    """tau_w = G h — from force balance ONLY, so rheology-independent.

    Identical to the Newtonian result. A solver disagreeing with this has its
    transport model wired in wrongly; it says nothing about the rheology.
    """
    return pressure_gradient * h


# --- Regularisation: what a nuMax cap does to the plug -----------------------
#
# OpenFOAM regularises the singular yield-stress viscosity as
#     nu(gammadot) = min(nuMax, [sqrt(tau0/gammadot) + sqrt(nu_c)]^2)
# so the cap engages where the unregularised nu reaches nuMax, i.e. at
#     gammadot_cap = tau0 / (sqrt(nuMax) - sqrt(nu_c))^2.
# Setting the exact shear rate equal to that and solving for y gives the
# closed forms below. The plug is never rigid: it creeps.


def cap_active_half_width(tau_y, pressure_gradient, nu_c, nu_max):
    """Half-width of the cap-active (nu == nuMax) region.

        y_cap = y_p / (1 - sqrt(nu_c/nuMax))^2
              ~ y_p (1 + 2 sqrt(nu_c/nuMax))

    ALWAYS WIDER than the true plug (positive bias), with relative error
    2 sqrt(nu_c/nuMax) — square-root, not linear, in the cap ratio, because
    the shear rate leaves the yield point quadratically.
    """
    y_p = plug_half_width(tau_y, pressure_gradient)
    return y_p / (1.0 - np.sqrt(nu_c / nu_max)) ** 2


def plug_width_relative_bias(nu_c, nu_max):
    """Leading-order relative plug-width bias eps = 2 sqrt(nu_c/nuMax)."""
    return 2.0 * np.sqrt(nu_c / nu_max)


def plug_velocity_variation(tau_y, pressure_gradient, nu_max):
    """Residual velocity variation across the capped plug.

    Inside the plug the stress is still tau = G y (force balance) while the
    viscosity is pinned at nuMax, so the creep shear rate is G y / nuMax and

        delta_u = integral_0^{y_p} G y / nuMax dy = G y_p^2 / (2 nuMax)

    which scales as 1/nuMax — a DIFFERENT exponent from the plug width, from
    the same cap.
    """
    y_p = plug_half_width(tau_y, pressure_gradient)
    return pressure_gradient * y_p**2 / (2.0 * nu_max)
