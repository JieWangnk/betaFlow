"""Taylor-Aris dispersion of a tracer in laminar pipe flow.

THE LAST EXACT ORACLE in this programme. Margination, capture efficiency and
deposition have no closed forms, so this is the final rung where a result can
be checked against truth rather than estimated.

The case carries FOUR independent exact anchors, three of them new in kind:

  1. Long time, t >> tau_r = a^2/D:   sigma_x^2 -> 2 D_eff t
     with D_eff = D + a^2 U^2 / (48 D).
  2. Short time, t << tau_r:          sigma_x^2 - 2 D t -> (U^2/3) t^2
     A DIFFERENT POWER OF t, so a code that lands D_eff correctly by accident
     still misses this.
  3. At all times, with and without flow:  P(r) = 2r/a^2
     Axial flow does not couple to r, so the radial distribution is an exact
     invariant independent of the physics under test. This is the gate.
  4. Size sweep:  R_cr = (2/(pi sqrt(3))) k_B T / (mu a U)
     Tests the SHAPE of D_eff(R) — falling below R_cr, rising above it — not
     a single point.

CITATIONS
  Taylor, G.I. (1953), Proc. R. Soc. A 219:186-203.
  Aris, R. (1956), Proc. R. Soc. A 235:67-77.
  Decuzzi, P., Causa, F., Ferrari, M. & Netti, P.A. (2006),
    Ann. Biomed. Eng. 34:633-641, Eq. 15 (D_eff) and Eq. 18 (R_cr).
  Einstein, A. (1905), Ann. Phys. 322(8):549-560 (Stokes-Einstein).
  The Decuzzi equation numbers were supplied by the user; their ALGEBRA is
  verified here rather than taken on trust — `verify_limits` checks that
  Eq. 15 equals Taylor-Aris with Stokes-Einstein substituted (6/48 = 1/8),
  and that Eq. 18 is the stationary point of Eq. 15 by differentiation.

VELOCITY STATISTICS, derived not assumed. For u(r) = 2U(1 - r^2/a^2) with
P(r) = 2r/a^2:
    E[u]   = int_0^a 2U(1-r^2/a^2)(2r/a^2) dr = U
    E[u^2] = int_0^a 4U^2(1-r^2/a^2)^2 (2r/a^2) dr = 4U^2/3
    Var(u) = 4U^2/3 - U^2 = U^2/3        exactly.
This U^2/3 is the short-time prefactor in anchor 2.

GEOMETRY-SPECIFIC. That constant is Poiseuille's alone: it comes from
int_0^1 4(1-s^2)^2 * 2s ds with P(r) = 2r/a^2. A plug, a power-law or a
Casson profile gives a different number, so the prefactor check is tied to
the velocity profile in the same way the wall-shear lever arm (h vs a/2) is
tied to the cross-section. Do not carry it to another profile.

MEASURED SHORT-TIME DEFICITS, and why they are physics. The fitted exponent
comes in at ~1.93 and the fitted prefactor at ~0.93 of U^2/3. BOTH are
deficits, and they share a sign because they are two symptoms of one
correction: at finite t radial diffusion has already begun decorrelating the
velocity a particle samples, so sigma_x^2 grows slower than t^2, which drags
the exponent below 2 AND the back-extrapolated prefactor below Var(u). Two
independent errors would not reliably share a sign. Verified by moving the
fit window: over the physics-dominated range both deficits shrink together
(exponent -0.081 -> -0.049, log-prefactor -0.091 -> -0.066 as the window
moves from [0.004, 0.02] to [0.002, 0.01] tau_r). They reverse at the two
earliest windows, but that is statistical rather than physical: there the
subtracted 2 D t is ~90% of sigma_x^2 so the residual carries ~10x amplified
noise, and quadrupling the ensemble recovers the trend (exponent 1.881 ->
1.945, prefactor 0.809 -> 0.901 at N = 6e4 -> 2.4e5, while the mid-window
barely moves).

WHEN THE ASYMPTOTIC REGIME ARRIVES. Radial relaxation in a cylinder with
no-flux walls has eigenfunctions J0(beta_n r/a) with J1(beta_n) = 0; the first
nonzero root is beta_1 = 3.8317, so the pre-asymptotic correction decays as
exp(-beta_1^2 t/tau_r) = exp(-14.68 t/tau_r) — 4e-7 at t = tau_r and 1e-13 at
2 tau_r. The asymptotic regime therefore arrives at ~tau_r, NOT at 50 tau_r,
which is why the runs here use t_max = 10 tau_r and fit over [2, 10] tau_r.
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import jn_zeros

from betaflow.analytic.brownian import BOLTZMANN, stokes_einstein

# First nonzero root of J1, setting the radial relaxation rate.
BESSEL_J1_FIRST_ROOT = float(jn_zeros(1, 1)[0])  # 3.8317059702


def velocity_profile(r_over_a, u_mean):
    """u(r) = 2 U (1 - (r/a)^2) — Poiseuille, mean U."""
    s = np.asarray(r_over_a, dtype=float)
    return 2.0 * u_mean * (1.0 - s**2)


def velocity_variance(u_mean):
    """Var(u) = U^2/3 exactly, over P(r) = 2r/a^2. See module docstring."""
    return u_mean**2 / 3.0


def radial_pdf(r, a):
    """P(r) = 2r/a^2 — uniform in the cross-section."""
    return 2.0 * np.asarray(r, dtype=float) / a**2


def radial_cdf(r_over_a):
    """F(r) = (r/a)^2. Used directly as the KS reference."""
    return np.clip(np.asarray(r_over_a, dtype=float), 0.0, 1.0) ** 2


def radial_relaxation_time(a, diffusivity):
    """tau_r = a^2 / D."""
    return a**2 / diffusivity


def asymptotic_onset(tolerance=1e-6):
    """t/tau_r at which the pre-asymptotic correction falls below `tolerance`.

    exp(-beta_1^2 t/tau_r) < tolerance. Returns ~0.63 at 1e-6, which is why
    fitting from 2 tau_r is generous rather than marginal.
    """
    return -np.log(tolerance) / BESSEL_J1_FIRST_ROOT**2


def d_eff(diffusivity, a, u_mean):
    """Taylor-Aris effective axial diffusivity D + a^2 U^2 / (48 D)."""
    return diffusivity + a**2 * u_mean**2 / (48.0 * diffusivity)


def d_eff_decuzzi(temperature, mu, particle_radius, a, u_mean):
    """Decuzzi et al. (2006) Eq. 15, in its published form.

        D_eff = k_B T/(6 pi mu R) + R_e^2 U^2 pi mu R / (8 k_B T)

    Identical to `d_eff` with Stokes-Einstein substituted, because
    a^2 U^2/(48 D) with D = k_B T/(6 pi mu R) gives
    a^2 U^2 * 6 pi mu R/(48 k_B T) = a^2 U^2 pi mu R/(8 k_B T), i.e. 6/48 = 1/8.
    """
    kt = BOLTZMANN * temperature
    return kt / (6.0 * np.pi * mu * particle_radius) + (
        a**2 * u_mean**2 * np.pi * mu * particle_radius / (8.0 * kt)
    )


def critical_radius(temperature, mu, a, u_mean):
    """Decuzzi et al. (2006) Eq. 18: R_cr = (2/(pi sqrt(3))) k_B T/(mu a U).

    The stationary point of Eq. 15. Verified by differentiation in
    `verify_limits`, not hardcoded as a coincidence.
    """
    return (2.0 / (np.pi * np.sqrt(3.0))) * BOLTZMANN * temperature / (mu * a * u_mean)


def peclet_at_critical_radius():
    """Pe = a U / D at R_cr is exactly sqrt(48).

    R_cr is where the two terms of D_eff are equal: D = a^2U^2/(48 D), so
    D = a U/sqrt(48) and Pe = sqrt(48) = 6.9282. A useful invariant: the
    sweep's midpoint should land here regardless of the physical parameters.
    """
    return np.sqrt(48.0)


def msd_axial_short(t, u_mean):
    """Shear-sampling (ballistic) limit: sigma_x^2 - 2 D t -> (U^2/3) t^2."""
    return velocity_variance(u_mean) * np.asarray(t, dtype=float) ** 2


def msd_axial_long(t, d_eff_value):
    """Asymptotic dispersion: sigma_x^2 -> 2 D_eff t."""
    return 2.0 * d_eff_value * np.asarray(t, dtype=float)


def verify_limits(rtol=1e-12):
    """Check the oracle's internal consistency before it is used as truth."""
    t_k, mu, a, u = 310.0, 1.0e-3, 10.0e-6, 3.1462e-6
    r_p = 150.0e-9
    errors = {}

    # 1. Eq. 15 == Taylor-Aris with Stokes-Einstein substituted (6/48 = 1/8).
    d = stokes_einstein(t_k, mu, r_p)
    errors["eq15_vs_taylor_aris"] = abs(
        d_eff_decuzzi(t_k, mu, r_p, a, u) / d_eff(d, a, u) - 1.0
    )

    # 2. Eq. 18 is the stationary point of Eq. 15 — found by differentiation,
    #    not assumed.
    # Analytic derivative of Eq. 15, written out independently rather than
    # rearranged towards Eq. 18 (a finite difference here is limited by its
    # own truncation at ~1e-11 and would not test the algebra):
    #   d/dR [ kT/(6 pi mu R) + a^2 U^2 pi mu R/(8 kT) ]
    #     = -kT/(6 pi mu R^2) + a^2 U^2 pi mu/(8 kT)
    def derivative(radius):
        kt = BOLTZMANN * t_k
        return -kt / (6.0 * np.pi * mu * radius**2) + (
            a**2 * u**2 * np.pi * mu / (8.0 * kt)
        )

    r_cr = critical_radius(t_k, mu, a, u)
    r_numeric = brentq(derivative, 0.1 * r_cr, 10.0 * r_cr, xtol=1e-18, rtol=8.9e-16)
    errors["eq18_vs_differentiation"] = abs(r_numeric / r_cr - 1.0)

    # 3. At R_cr the Peclet number is exactly sqrt(48).
    pe = a * u / stokes_einstein(t_k, mu, r_cr)
    errors["peclet_at_r_cr"] = abs(pe / peclet_at_critical_radius() - 1.0)

    # 4. Velocity statistics by quadrature against the closed forms.
    s = np.linspace(0.0, 1.0, 2000001)
    pdf = 2.0 * s
    u_of_s = velocity_profile(s, u)
    mean = np.trapezoid(u_of_s * pdf, s)
    second = np.trapezoid(u_of_s**2 * pdf, s)
    errors["velocity_mean"] = abs(mean / u - 1.0)
    errors["velocity_variance"] = abs((second - mean**2) / velocity_variance(u) - 1.0)

    for name, err in errors.items():
        limit = rtol if "velocity" not in name else 1e-9
        if not err < limit:
            raise AssertionError(f"taylor_aris oracle {name} error {err:.3e} > {limit:.0e}")
    return {k: float(v) for k, v in errors.items()}
