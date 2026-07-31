"""Oracle self-verification — pure analytic checks, NO solver required.

Every oracle in the repo is ground truth for a case, so each one is itself
verified here against independent analytic facts: exact limits, closed-form
identities, and differentiated forms. This is the "verify the verifier" tier.

It runs in seconds with no OpenFOAM installed, which makes it the tier CI can
run on every push regardless of environment.
"""

import numpy as np
import pytest

from betaflow.analytic import (
    brownian,
    carreau,
    casson,
    couette,
    pipe,
    poiseuille,
    womersley,
)

pytestmark = pytest.mark.oracle


def test_poiseuille_relations():
    # u_mean = (2/3) u_max exactly, and the profile is the unit parabola.
    g, mu, h = 0.4, 0.02, 1.0
    assert poiseuille.u_mean(g, mu, h) == pytest.approx(
        (2.0 / 3.0) * poiseuille.u_max(g, mu, h), rel=1e-15
    )
    y = np.linspace(-1.0, 1.0, 21)
    np.testing.assert_allclose(poiseuille.velocity_profile(y), 1.0 - y**2, rtol=1e-15)
    # tau_w = G h, and the pressure gradient inverts the bulk relation.
    assert poiseuille.tau_wall(g, h) == pytest.approx(g * h, rel=1e-15)
    u_bar = poiseuille.u_mean(g, mu, h)
    assert poiseuille.pressure_gradient(u_bar, h, mu) == pytest.approx(g, rel=1e-13)
    # Re definition round-trips through the stated convention.
    nu = 0.02
    assert poiseuille.reynolds(u_bar, h, nu) == pytest.approx(u_bar * 2 * h / nu, rel=1e-15)


def test_couette_relations():
    y = np.linspace(0.0, 1.0, 21)
    np.testing.assert_allclose(couette.velocity_profile(y), y, rtol=1e-15)
    assert couette.tau_wall(0.02, 1.5, 0.5) == pytest.approx(0.02 * 1.5 / 0.5, rel=1e-15)


def test_casson_differentiation_identity():
    """-du/dy must equal (sqrt(G y) - sqrt(tau_y))^2 / mu_c outside the plug."""
    g, tau_y, mu, h = 0.4, 0.08, 0.02, 1.0
    y = np.linspace(0.25, 0.99, 9)
    d = 1e-6
    numerical = -(
        casson.velocity(y + d, g, tau_y, mu, h) - casson.velocity(y - d, g, tau_y, mu, h)
    ) / (2 * d)
    np.testing.assert_allclose(
        numerical, casson.shear_rate(y, g, tau_y, mu), rtol=1e-5
    )
    # No-slip, and the plug is flat.
    assert float(casson.velocity(h, g, tau_y, mu, h)) == pytest.approx(0.0, abs=1e-15)
    y_p = casson.plug_half_width(tau_y, g)
    assert float(casson.velocity(0.5 * y_p, g, tau_y, mu, h)) == pytest.approx(
        float(casson.velocity(0.0, g, tau_y, mu, h)), rel=1e-15
    )


def test_casson_newtonian_limit():
    """xi -> 0 must recover the Poiseuille parabola."""
    assert casson.u_max_over_U0(1e-14) == pytest.approx(0.5, rel=1e-6)
    assert casson.u_mean_over_U0(1e-14) == pytest.approx(1.0 / 3.0, rel=1e-6)


def test_womersley_identity_and_limit():
    """tau_hat = h(G - i w <u>) in normalised form, and the alpha -> 0 limit."""
    for alpha in (0.5, 5.0, 20.0):
        # In units of tau_ref = G h and u_ref = G/omega, the balance reads
        # tau_hat = 1 - i <uhat>; compare with the closed form tanh(K)/K.
        balance = 1.0 - 1.0j * womersley.complex_bulk(alpha)
        assert womersley.complex_wall_shear(alpha) == pytest.approx(balance, rel=1e-12)
    # Quasi-steady limit: tau -> G h and the profile -> the parabola.
    assert womersley.complex_wall_shear(1e-4).real == pytest.approx(1.0, rel=1e-8)
    y = np.linspace(-1.0, 1.0, 11)
    profile = womersley.complex_profile(y, 1e-3)
    parabola = 1.0 - y**2
    np.testing.assert_allclose(
        np.abs(profile) / np.max(np.abs(profile)), parabola, rtol=1e-6
    )


def test_carreau_self_verification():
    """Newtonian and power-law limits, to machine precision."""
    errors = carreau.verify_limits(rtol=1e-12)
    assert errors["newtonian_limit"] < 1e-12
    assert errors["power_law_limit"] < 1e-12


def test_carreau_stress_balance():
    """The rootfind must satisfy nu(gammadot)*gammadot = G y exactly."""
    g, nu0, nu_inf, k, n = 0.4, 0.02, 2e-4, 1.0, 0.5
    for y in (0.05, 0.25, 0.5, 1.0):
        gdot = carreau.shear_rate(g * y, nu0, nu_inf, k, n)
        tau = carreau.viscosity(gdot, nu0, nu_inf, k, n) * gdot
        assert tau == pytest.approx(g * y, rel=1e-13)


def test_carreau_reduces_to_poiseuille():
    """With no thinning (k -> 0) the Carreau oracle IS the Poiseuille oracle."""
    g, nu0, h = 0.4, 0.02, 1.0
    y = np.linspace(0.0, h, 9)
    np.testing.assert_allclose(
        carreau.velocity(y, g, h, nu0, 0.0, 1e-14, 0.5),
        poiseuille.velocity_profile(y / h) * poiseuille.u_max(g, nu0, h),
        rtol=1e-12,
    )


def test_pipe_force_balance_and_kernels():
    """tau(r) = G r / 2, and the pipe/channel kernels are NOT interchangeable."""
    g, a = 0.4, 1.0
    assert pipe.tau_wall(g, a) == pytest.approx(g * a / 2.0, rel=1e-15)
    assert pipe.poiseuille_u_max(g, a, 0.02) / pipe.poiseuille_u_mean(g, a, 0.02) == (
        pytest.approx(2.0, rel=1e-15)
    )
    # Womersley pipe: tauhat = (a/2)(G - i w <u>), in normalised units
    # tauhat/tau_ref = 1 - i <uhat>/u_ref.
    for al in (0.5, 5.0, 20.0):
        assert pipe.womersley_wall_shear(al) == pytest.approx(
            1.0 - 1.0j * pipe.womersley_bulk(al), rel=1e-12
        )
    # Plug radius r_p = 2 tau_y / G — twice the channel's tau_y / G.
    assert pipe.plug_radius(0.05, g) == pytest.approx(2 * 0.05 / g, rel=1e-15)
    # The two geometries' pulsatile kernels must differ substantially in wall
    # shear even where their profiles nearly agree (see README).
    j0, cosh = pipe.womersley_wall_shear(10.0), womersley.complex_wall_shear(10.0)
    assert abs(abs(cosh) - abs(j0)) / abs(j0) > 0.4


def test_brownian_fluctuation_dissipation():
    """The noise and the drag must share one friction coefficient."""
    out = brownian.verify_fluctuation_dissipation(310.0, 3.5e-3, 50e-9)
    assert out["round_trip_error"] == 0.0
    # D = k_B T / zeta, and MSD is 6 D t with 2 D t per component.
    d = brownian.stokes_einstein(310.0, 3.5e-3, 50e-9)
    assert d == pytest.approx(
        brownian.BOLTZMANN * 310.0 / brownian.friction_coefficient(3.5e-3, 50e-9),
        rel=1e-15,
    )
    t = np.array([0.0, 0.5, 1.0])
    np.testing.assert_allclose(brownian.msd(t, d), 3 * brownian.msd_per_component(t, d),
                               rtol=1e-15)
    # Overdamped limit is justified only if St << 1.
    assert brownian.stokes_number(1050.0, 50e-9, 3.5e-3, 0.3, 0.01) < 1e-6
