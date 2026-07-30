"""Analytic oracle for steady Carreau(-Yasuda) flow in a plane channel.

Geometry: plane channel, half-height h, no-slip walls at y = +/- h, constant
streamwise pressure gradient G = -dp/dx > 0.

THIS CASE HAS AN EXACT SOLUTION. It is sometimes claimed that shear-thinning
channel flow is "solver-only", requiring GCI-style solution verification.
That is wrong. In steady fully-developed 1-D channel flow the momentum
equation integrates once to

    tau(y) = G * y            EXACTLY, for ANY rheology

— the same force balance that makes tau_w = G h rheology-independent. So for
any generalised Newtonian model nu(gammadot) the solution follows in two
steps, each to machine precision:

  1. Pointwise rootfind: at each y solve the scalar, monotone equation
         nu(gammadot) * gammadot = G * y
     for gammadot. The left side is strictly increasing in gammadot for
     n > 0 (it behaves as gammadot^n at large strain rate), so the root is
     unique and bracketing is trivial.
  2. Quadrature: u(y) = integral_y^h gammadot(y') dy', using u(h) = 0.
     gammadot is smooth (no yield stress, no kink) and vanishes linearly at
     the centreline, so adaptive quadrature reaches machine precision.

There is no ODE integration and no shooting; nothing here is a numerical
approximation of the physics, only of the quadrature, at ~1e-14. This is
therefore CODE verification with an order-of-accuracy test, not solution
verification with GCI.

Carreau-Yasuda law, kinematic (OpenFOAM's BirdCarreau coefficient names
exactly — nu0, nuInf, k, n, with a defaulting to 2):

    nu(gammadot) = nuInf + (nu0 - nuInf) (1 + (k gammadot)^a)^((n - 1)/a)

CARREAU NUMBER DEFINITION — state it, as with Re, or invite silent
disagreement:

    Cu = k * (G h / nu0)

    * k is the model time constant [s]
    * G h / nu0 is the NEWTONIAN-LIMIT WALL SHEAR RATE: the wall stress
      tau_w = G h divided by the ZERO-shear viscosity nu0.

So Cu is "how far into shear-thinning the wall would be if the fluid stayed
Newtonian". It is defined from nu0 (not nuInf), from the WALL shear rate (not
a mean or centreline rate), and it uses the discrete G the solver actually
applied when computed from a run. Alternatives seen elsewhere: k times a
bulk rate u_mean/h, or k times the true (thinned) wall rate — both differ
from this by an O(1) factor that grows with Cu.

Limits the oracle must reproduce (asserted in `verify_limits`):
    Cu -> 0            : parabola with viscosity nu0
    Cu -> inf, nuInf=0 : power law, u ~ h^((n+1)/n) - |y|^((n+1)/n)

UNITS: kinematic in, kinematic out (nu0, nuInf [m^2/s]; G [m/s^2]; tau
[m^2/s^2]), matching incompressible OpenFOAM and the rest of betaflow.
"""

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq


def viscosity(gammadot, nu0, nu_inf, k, n, a=2.0):
    """Carreau-Yasuda kinematic viscosity at a given strain rate."""
    g = np.asarray(gammadot, dtype=float)
    return nu_inf + (nu0 - nu_inf) * (1.0 + (k * g) ** a) ** ((n - 1.0) / a)


def shear_rate(tau, nu0, nu_inf, k, n, a=2.0):
    """Solve nu(gammadot)*gammadot = tau for gammadot (scalar, exact to ~1e-15).

    Monotone in gammadot, so bracketing is safe:
      * nu <= nu0        =>  root >= tau/nu0
      * nu >= nu_inf     =>  root <= tau/nu_inf   (when nu_inf > 0)
    With nu_inf = 0 the upper bracket is grown from the power-law estimate
    until it straddles the root.
    """
    tau = float(tau)
    if tau <= 0.0:
        return 0.0

    def residual(g):
        return viscosity(g, nu0, nu_inf, k, n, a) * g - tau

    # For shear thinning (n < 1) the Carreau factor is <= 1, so nu <= nu0 and
    # tau/nu0 is a lower bound on the root. It is attained exactly in the
    # Newtonian limit (k -> 0), where the bracket would otherwise collapse
    # onto a double root and brentq would have nothing to bisect.
    lo = tau / nu0
    if residual(lo) >= 0.0:
        return lo
    if nu_inf > 0.0:
        hi = tau / nu_inf  # nu >= nu_inf, so this brackets from above
    else:
        # Power-law branch: nu ~ nu0 k^(n-1) g^(n-1)  =>  tau ~ nu0 k^(n-1) g^n
        hi = max(2.0 * lo, (tau / (nu0 * k ** (n - 1.0))) ** (1.0 / n)) if k > 0 else 2.0 * lo
        while residual(hi) < 0.0:
            hi *= 2.0
    return brentq(residual, lo, hi, xtol=1e-16, rtol=8.9e-16, maxiter=200)


def _shear_rate_profile(y, pressure_gradient, nu0, nu_inf, k, n, a=2.0):
    return np.array(
        [
            shear_rate(pressure_gradient * abs(float(yy)), nu0, nu_inf, k, n, a)
            for yy in np.atleast_1d(y)
        ]
    )


def velocity(y, pressure_gradient, h, nu0, nu_inf, k, n, a=2.0):
    """u(y) = integral_|y|^h gammadot dy', with u(+/-h) = 0."""
    out = []
    for yy in np.atleast_1d(np.asarray(y, dtype=float)):
        value, _ = quad(
            lambda s: shear_rate(pressure_gradient * s, nu0, nu_inf, k, n, a),
            abs(yy),
            h,
            epsabs=1e-14,
            epsrel=1e-13,
            limit=200,
        )
        out.append(value)
    return np.array(out)


def u_max(pressure_gradient, h, nu0, nu_inf, k, n, a=2.0):
    """Centreline velocity u(0)."""
    return float(velocity(0.0, pressure_gradient, h, nu0, nu_inf, k, n, a)[0])


def u_mean(pressure_gradient, h, nu0, nu_inf, k, n, a=2.0):
    """Bulk velocity (1/h) integral_0^h u dy, by symmetry."""
    value, _ = quad(
        lambda yy: velocity(yy, pressure_gradient, h, nu0, nu_inf, k, n, a)[0],
        0.0,
        h,
        epsabs=1e-13,
        epsrel=1e-12,
        limit=200,
    )
    return value / h


def velocity_profile(y_over_h, pressure_gradient, h, nu0, nu_inf, k, n, a=2.0):
    """Non-dimensional u/u_max at Y = y/h — the harness oracle signature."""
    y = np.asarray(y_over_h, dtype=float) * h
    return velocity(y, pressure_gradient, h, nu0, nu_inf, k, n, a) / u_max(
        pressure_gradient, h, nu0, nu_inf, k, n, a
    )


def tau_wall(pressure_gradient, h):
    """tau_w = G h — force balance only, so rheology-independent."""
    return pressure_gradient * h


def carreau_number(k, pressure_gradient, h, nu0):
    """Cu = k * (G h / nu0). See the module docstring before substituting."""
    return k * pressure_gradient * h / nu0


def pressure_gradient_for_bulk(u_mean_target, h, nu0, nu_inf, k, n, a=2.0):
    """G giving the requested bulk velocity, with the rheology held fixed."""

    def residual(g):
        return u_mean(g, h, nu0, nu_inf, k, n, a) - u_mean_target

    # Shear thinning only ever raises the flow rate relative to Newtonian, so
    # the Newtonian G is an upper bound; bracket downward from it.
    hi = 3.0 * nu0 * u_mean_target / h**2
    lo = hi
    while residual(lo) > 0.0:
        lo /= 2.0
    return brentq(residual, lo, hi, xtol=1e-15, rtol=8.9e-16)


def drive_for_carreau_number(u_mean_target, h, nu0, nu_inf, n, cu, a=2.0):
    """(G, k) realising both the requested bulk velocity and Carreau number.

    Cu = k G h / nu0 couples the two: k depends on G, and G depends on k.
    Substituting k = Cu nu0 / (G h) leaves one equation in G.
    """

    def residual(g):
        k = cu * nu0 / (g * h)
        return u_mean(g, h, nu0, nu_inf, k, n, a) - u_mean_target

    hi = 3.0 * nu0 * u_mean_target / h**2
    lo = hi
    while residual(lo) > 0.0:
        lo /= 2.0
    g = brentq(residual, lo, hi, xtol=1e-15, rtol=8.9e-16)
    return g, cu * nu0 / (g * h)


# --- oracle self-verification ------------------------------------------------


def verify_limits(rtol=1e-12):
    """Check the oracle against its two analytic limits. Verify the verifier.

    Returns a dict of relative errors; raises if either exceeds rtol.
    """
    h, nu0, g_grad = 1.0, 0.02, 0.4
    y = np.linspace(0.0, h, 11)

    # 1. Cu -> 0: Newtonian parabola with nu0. Reached with a tiny time
    #    constant, where (k gammadot)^a -> 0 and nu -> nu0.
    newtonian = g_grad * (h**2 - y**2) / (2.0 * nu0)
    got = velocity(y, g_grad, h, nu0, 0.0, 1e-12, 0.5)
    err_newtonian = float(np.max(np.abs(got - newtonian) / np.max(newtonian)))

    # 2. Cu -> inf with nuInf = 0: power law with consistency nu0 k^(n-1),
    #    u = (G/m)^(1/n) n/(n+1) (h^((n+1)/n) - y^((n+1)/n)).
    n = 0.5
    k = 1e8
    m_pl = nu0 * k ** (n - 1.0)
    p = (n + 1.0) / n
    power_law = (g_grad / m_pl) ** (1.0 / n) * (n / (n + 1.0)) * (h**p - y**p)
    got = velocity(y, g_grad, h, nu0, 0.0, k, n)
    err_power_law = float(np.max(np.abs(got - power_law) / np.max(power_law)))

    errors = {"newtonian_limit": err_newtonian, "power_law_limit": err_power_law}
    for name, err in errors.items():
        if not err < rtol:
            raise AssertionError(f"oracle {name} error {err:.3e} exceeds {rtol:.0e}")
    return errors
