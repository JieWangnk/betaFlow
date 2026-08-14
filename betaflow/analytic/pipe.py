"""Analytic references for steady and pulsatile flow in a CIRCULAR PIPE.

Geometry: circular pipe of radius a, no-slip at r = a, axis at r = 0, driven
by a streamwise pressure gradient G = -dp/dx > 0.

THE ONE CHANGE FROM THE CHANNEL, from which everything else follows. A force
balance on a cylinder of radius r gives (pi r^2) G = (2 pi r) tau, so

    tau(r) = G r / 2        EXACTLY, for ANY rheology
    tau_w  = G a / 2

against the plane channel's tau(y) = G y and tau_w = G h. The factor of two
propagates into every profile below; it is the single most likely place for a
silent geometry error, which is why the conservation identity is written
against G a / 2 and asserted per case.

REYNOLDS NUMBER DEFINITION (stated, as for the channel, to prevent silent
disagreement):

    Re = u_mean * (2 a) / nu = u_mean * D / nu

    * velocity scale: BULK (area-averaged) velocity
    * length scale:   the pipe DIAMETER D = 2a

This is the standard pipe convention, under which laminar flow persists to
Re ~ 2300 (Reynolds 1883). It is dimensionally the same rule as the channel's
Re = u_mean * 2h / nu — "bulk velocity times full transverse extent" — so the
two are consistent, but a is a RADIUS where h was a HALF-HEIGHT.

WOMERSLEY NUMBER DEFINITION:

    alpha = a * sqrt(omega / nu)

    * length scale: the pipe RADIUS a (Womersley 1955)

CITATIONS
  Poiseuille profile and Hagen-Poiseuille law:
      Batchelor, G.K. (1967), An Introduction to Fluid Dynamics, CUP, §4.2.
      Sutera, S.P. & Skalak, R. (1993), "The history of Poiseuille's law",
      Annu. Rev. Fluid Mech. 25:1-19.
  Pulsatile (Womersley) pipe flow, J0 Bessel kernel:
      Womersley, J.R. (1955), "Method for the calculation of velocity, rate
      of flow and viscous drag in arteries when the pressure gradient is
      known", J. Physiol. 127(3):553-563.
  Casson constitutive law:
      Casson, N. (1959), "A flow equation for pigment-oil suspensions of the
      printing ink type", in Rheology of Disperse Systems (ed. C.C. Mill),
      Pergamon, pp. 84-104.
  Casson blood flow in tubes, and the plug:
      Fung, Y.C. (1997), Biomechanics: Circulation, 2nd ed., Springer, ch. 3.
  Plug-to-vessel radius ratio named xi_c:
      attribution supplied by the user as Gentile et al. (2008). NOT
      independently verified here — see the note on `plug_radius_ratio`. The
      FORMULA itself needs no citation: it follows from tau(r_p) = tau_y with
      tau(r) = G r / 2, and is verified in tests by differentiation.

GEOMETRY KERNEL WARNING. The pulsatile kernel here is the J0 Bessel function
of complex argument, which is correct for the CIRCULAR PIPE. The plane
channel's kernel is a complex cosh (see betaflow/analytic/womersley.py).
These are not interchangeable, and swapping them is a silent O(1) error that
still produces a plausible-looking profile.

UNITS: kinematic in, kinematic out (nu [m^2/s], G [m/s^2], tau [m^2/s^2]),
matching incompressible OpenFOAM and the rest of betaflow.
"""

import numpy as np
from scipy.special import jv

# ---------------------------------------------------------------------------
# Force balance — rheology-independent (Batchelor 1967, §4.2)
# ---------------------------------------------------------------------------


def tau_of_r(r, pressure_gradient):
    """tau(r) = G r / 2. True for ANY rheology; pure force balance."""
    return pressure_gradient * np.asarray(r, dtype=float) / 2.0


def tau_wall(pressure_gradient, a):
    """tau_w = G a / 2 — the pipe analogue of the channel's G h."""
    return pressure_gradient * a / 2.0


def reynolds(u_mean, a, nu):
    """Re = u_mean * (2a) / nu — bulk velocity, pipe DIAMETER."""
    return u_mean * 2.0 * a / nu


def womersley_number(a, omega, nu):
    """alpha = a sqrt(omega/nu) — pipe RADIUS (Womersley 1955)."""
    return a * np.sqrt(omega / nu)


# ---------------------------------------------------------------------------
# Steady Newtonian (Hagen-Poiseuille)
# ---------------------------------------------------------------------------


def poiseuille_velocity(r, pressure_gradient, a, mu):
    """u(r) = (G/4mu)(a^2 - r^2)  [Batchelor 1967 §4.2].

    Note the 1/4: the channel is (G/2mu)(h^2 - y^2). The factor traces
    directly to tau(r) = G r / 2 versus tau(y) = G y.
    """
    r = np.asarray(r, dtype=float)
    return pressure_gradient / (4.0 * mu) * (a**2 - r**2)


def poiseuille_profile(r_over_a):
    """Non-dimensional u/u_max = 1 - (r/a)^2 — the framework analytic reference signature."""
    r = np.asarray(r_over_a, dtype=float)
    return 1.0 - r**2


def poiseuille_u_max(pressure_gradient, a, mu):
    """u_max = G a^2 / (4 mu), at the axis."""
    return pressure_gradient * a**2 / (4.0 * mu)


# u_mean = u_max / 2 for a pipe (the channel gives 2/3). Integrating the
# paraboloid: (2/a^2) int_0^a (1-(r/a)^2) r dr = 1/2.
POISEUILLE_U_MAX_OVER_U_MEAN = 2.0


def poiseuille_u_mean(pressure_gradient, a, mu):
    """u_mean = u_max/2 = G a^2/(8 mu) — the Hagen-Poiseuille law."""
    return poiseuille_u_max(pressure_gradient, a, mu) / POISEUILLE_U_MAX_OVER_U_MEAN


def poiseuille_pressure_gradient(u_mean, a, mu):
    """G giving the requested bulk velocity: 8 mu u_mean / a^2."""
    return 8.0 * mu * u_mean / a**2


# ---------------------------------------------------------------------------
# Pulsatile (Womersley 1955) — J0 kernel, CIRCULAR PIPE ONLY
# ---------------------------------------------------------------------------


def _beta(alpha):
    """beta = i^{3/2} alpha, the complex Bessel argument (Womersley 1955).

    Equivalent to alpha (i-1)/sqrt(2); J0 is even so the sign is immaterial.
    """
    return alpha * (1.0j - 1.0) / np.sqrt(2.0)


def womersley_profile(r_over_a, alpha):
    """Complex velocity uhat / u_ref at r/a, with u_ref = G/omega.

    uhat = (G/(i omega)) [1 - J0(beta r/a) / J0(beta)]     (Womersley 1955)

    so uhat/u_ref = -i [1 - J0(beta r/a)/J0(beta)]. Amplitude is abs(),
    phase (radians, relative to the G cos(omega t) forcing) is angle().
    """
    r = np.asarray(r_over_a, dtype=complex)
    b = _beta(alpha)
    return -1.0j * (1.0 - jv(0, b * r) / jv(0, b))


def womersley_bulk(alpha):
    """Complex bulk velocity <uhat>/u_ref = -i[1 - 2 J1(beta)/(beta J0(beta))].

    Uses int_0^x J0(t) t dt = x J1(x).
    """
    b = _beta(alpha)
    return -1.0j * (1.0 - 2.0 * jv(1, b) / (b * jv(0, b)))


def womersley_wall_shear(alpha):
    """Complex wall shear tauhat / tau_ref, tau_ref = G a / 2.

        tauhat/tau_ref = (2/beta) J1(beta)/J0(beta)

    Derived from -nu duhat/dr at r=a, and identically equal to the momentum
    balance (a/2)(G - i omega <u>) — the same conservation identity that
    holds for the channel, with a/2 in place of h. As alpha -> 0 it tends to
    1 (quasi-steady tau_w = G a / 2).
    """
    b = _beta(alpha)
    return 2.0 * jv(1, b) / (b * jv(0, b))


# ---------------------------------------------------------------------------
# Steady Casson in a pipe (Casson 1959 law; Fung 1997 ch. 3 for blood in tubes)
# ---------------------------------------------------------------------------


def plug_radius(tau_y, pressure_gradient):
    """Plug radius r_p = 2 tau_y / G, where tau(r_p) = tau_y.

    Follows directly from tau(r) = G r / 2. The channel analogue is
    y_p = tau_y / G — again the factor of two.
    """
    return 2.0 * tau_y / pressure_gradient


def plug_radius_ratio(tau_y, pressure_gradient, a):
    """xi_c = r_p / a = 2 tau_y / (G a) — the plug-to-vessel radius ratio.

    The NAME xi_c and its attribution to Gentile et al. (2008) were supplied
    by the user and are NOT independently verified here; they are recorded so
    the particle-transport work can consume the same symbol. The QUANTITY
    needs no attribution — it is tau(r_p) = tau_y under tau(r) = G r / 2, and
    is checked by differentiation in the tests.
    """
    return plug_radius(tau_y, pressure_gradient) / a


def casson_shear_rate(r, pressure_gradient, tau_y, mu_c):
    """-du/dr = (sqrt(G r/2) - sqrt(tau_y))^2 / mu_c outside the plug, else 0.

    Casson (1959): sqrt(tau) = sqrt(tau_y) + sqrt(mu_c gammadot), with
    tau = G r / 2.
    """
    r = np.abs(np.asarray(r, dtype=float))
    excess = np.sqrt(pressure_gradient * r / 2.0) - np.sqrt(tau_y)
    return np.where(excess > 0.0, excess**2 / mu_c, 0.0)


def casson_velocity(r, pressure_gradient, tau_y, mu_c, a):
    """Casson velocity profile in a pipe.

    Integrating the shear rate from r to a with u(a) = 0:

        u(r) = (1/mu_c) [ G (a^2 - r^2)/4
                          - (4/3) sqrt(G tau_y / 2) (a^1.5 - r^1.5)
                          + tau_y (a - r) ]

    for r_p <= r <= a, and u = u(r_p) inside the plug. Compare the channel:
    G/2 becomes G/4 and sqrt(G tau_y) becomes sqrt(G tau_y / 2).
    """
    r = np.abs(np.asarray(r, dtype=float))
    r_p = plug_radius(tau_y, pressure_gradient)
    rc = np.maximum(np.minimum(r, a), r_p)
    return (
        pressure_gradient * (a**2 - rc**2) / 4.0
        - (4.0 / 3.0) * np.sqrt(pressure_gradient * tau_y / 2.0) * (a**1.5 - rc**1.5)
        + tau_y * (a - rc)
    ) / mu_c


def casson_profile(r_over_a, xi_c):
    """Non-dimensional u/u_max at r/a for plug ratio xi_c — analytic reference signature."""
    r = np.asarray(r_over_a, dtype=float)
    # Work in units where a = 1, G = 1, mu_c = 1; then tau_y = xi_c/2.
    tau_y = xi_c / 2.0
    u = casson_velocity(r, 1.0, tau_y, 1.0, 1.0)
    return u / casson_velocity(0.0, 1.0, tau_y, 1.0, 1.0)


def casson_u_mean(pressure_gradient, tau_y, mu_c, a, n_quad=20001):
    """Bulk velocity (2/a^2) int_0^a u r dr, by quadrature.

    The integrand is smooth (the profile is C^1 at the plug edge, where the
    shear rate vanishes quadratically), so the trapezoid rule on a fine
    uniform grid is accurate to ~1e-12 and needs no adaptive machinery.
    """
    r = np.linspace(0.0, a, n_quad)
    u = casson_velocity(r, pressure_gradient, tau_y, mu_c, a)
    return float(2.0 / a**2 * np.trapezoid(u * r, r))


def casson_pressure_gradient_for_bulk(u_mean_target, a, mu_c, xi_c):
    """G realising the requested bulk velocity at plug ratio xi_c.

    xi_c = 2 tau_y/(G a) couples tau_y to G, so substituting
    tau_y = xi_c G a / 2 leaves one monotone equation in G, solved by
    bisection on a bracket anchored at the Newtonian value.
    """
    from scipy.optimize import brentq

    def residual(g):
        return casson_u_mean(g, xi_c * g * a / 2.0, mu_c, a) - u_mean_target

    lo = poiseuille_pressure_gradient(u_mean_target, a, mu_c)
    hi = lo
    while residual(hi) < 0.0:
        hi *= 2.0
    return brentq(residual, lo, hi, xtol=1e-15, rtol=8.9e-16)
