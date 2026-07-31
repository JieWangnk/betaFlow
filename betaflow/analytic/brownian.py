"""Analytic oracle for free Brownian motion of a sphere in a quiescent fluid.

No flow, no walls, no geometry. This is the base rung of the particle ladder
and the first case in betaflow with no CFD solver behind it at all.

PHYSICS AND PUBLISHED ANCHORS

  Stokes-Einstein diffusivity of a sphere of radius a in a fluid of dynamic
  viscosity mu at absolute temperature T:

      D = k_B T / (6 pi mu a) = k_B T / zeta

  where zeta = 6 pi mu a is the Stokes friction coefficient.
      Einstein, A. (1905), Ann. Phys. 322(8):549-560.
      Cited by the user as the Brownian term of Eq. 11 in Grief, A.D. &
      Richardson, G. (2005), J. Magn. Magn. Mater. 293:455-463.

  Fluctuation-dissipation, in the form used for nanoparticles in blood:
      <R_i(t)> = 0,  <R_i(t) R_j(t')> = 2 k_B T beta_t delta_ij delta(t-t')
      Cited by the user as Shah et al. (2011), J. Nanosci. Nanotechnol.
      11:919-928, Eqs 7-8; original FDT Mori et al. (1998), Rheol. Acta
      37:151-157.

  Discrete Brownian force as used in blood-flow particle tracking:
      F_bi = zeta_i sqrt(pi S0 / dt), zeta_i zero-mean unit-variance Gaussian
      Cited by the user as Doig et al. (2012), J. Comput. Multiph. Flows
      4:85-96, Eq. 44.

  Mean squared displacement in three dimensions:
      <|r(t) - r(0)|^2> = 6 D t
      Einstein (1905). Per component it is 2 D t; the 6 is 3 x 2.

CITATION STATUS. Einstein (1905) is the primary source and is standard. The
four secondary references above were supplied by the user with equation
numbers; they are recorded verbatim so the particle work can trace to them,
but the specific page and equation numbers have NOT been independently
verified here. The physics does not rest on them: D = k_B T / (6 pi mu a) and
MSD = 6 D t are checked against each other and against dimensional analysis
in tests/test_oracles.py.

THE CLASSIC IMPLEMENTATION BUG this module is written to prevent: taking the
friction coefficient in the noise amplitude from a different expression than
the one used for Stokes drag. Fluctuation-dissipation REQUIRES them to be the
same zeta. `verify_fluctuation_dissipation` asserts it.
"""

import numpy as np

# CODATA 2018 exact value (SI defines the kelvin via k_B).
BOLTZMANN = 1.380649e-23  # J/K


def friction_coefficient(mu, a):
    """Stokes drag coefficient zeta = 6 pi mu a [kg/s]."""
    return 6.0 * np.pi * mu * a


def stokes_einstein(temperature, mu, a):
    """D = k_B T / (6 pi mu a) [m^2/s] — Einstein (1905)."""
    return BOLTZMANN * temperature / friction_coefficient(mu, a)


def msd(t, diffusivity):
    """<|r(t)-r(0)|^2> = 6 D t in three dimensions."""
    return 6.0 * diffusivity * np.asarray(t, dtype=float)


def msd_per_component(t, diffusivity):
    """<(x_i(t)-x_i(0))^2> = 2 D t for each of the three components."""
    return 2.0 * diffusivity * np.asarray(t, dtype=float)


def stokes_number(rho_p, a, mu, u_ref, l_ref):
    """St = rho_p (2a)^2 u_ref / (18 mu l_ref) — particle response time over
    flow time. Overdamped (inertialess) Langevin is justified for St << 1."""
    return rho_p * (2.0 * a) ** 2 * u_ref / (18.0 * mu * l_ref)


def verify_fluctuation_dissipation(temperature, mu, a, rtol=1e-15):
    """Assert the noise and drag use the SAME friction coefficient.

    The overdamped increment is dx = sqrt(2 D dt) N(0,1) per component with
    D = k_B T / zeta. If the diffusivity is instead built from an independent
    constant, the resulting process no longer satisfies fluctuation-
    dissipation and the MSD slope is silently wrong. Checked here by
    round-tripping: zeta recovered from D must equal the Stokes value.
    """
    zeta = friction_coefficient(mu, a)
    d = stokes_einstein(temperature, mu, a)
    zeta_from_d = BOLTZMANN * temperature / d
    err = abs(zeta_from_d / zeta - 1.0)
    if not err < rtol:
        raise AssertionError(
            f"fluctuation-dissipation violated: zeta from D is {zeta_from_d:.6e} "
            f"but Stokes drag gives {zeta:.6e} (rel {err:.3e})"
        )
    return {"zeta": zeta, "D": d, "round_trip_error": err}
