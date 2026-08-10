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
    advection_diffusion,
    numerical_diffusion,
    brownian,
    carreau,
    casson,
    couette,
    pipe,
    poiseuille,
    taylor_aris,
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


def test_advection_diffusion_self_verification():
    """The Eulerian scalar oracle, and its cross-check against taylor_aris.

    `verify_limits` raises on any failure, so calling it IS the test. The
    assertions below restate the results a reader would want named.
    """
    ad = advection_diffusion
    out = ad.verify_limits()
    assert out, "verify_limits returned no checks"

    # Both constants re-derived numerically from the cell problem, not
    # compared against themselves.
    assert ad.dispersion_factor_numeric("pipe") == pytest.approx(1 / 48, rel=1e-8)
    assert ad.dispersion_factor_numeric("channel") == pytest.approx(2 / 105, rel=1e-8)

    # POSITIVE CONTROL for the whole route: the channel answer is only
    # trustworthy because the identical code path reproduces Aris's pipe value.
    assert ad.DISPERSION_FACTOR["pipe"] == pytest.approx(1 / 48, rel=1e-15)

    # Cross-module. Two independent implementations of the same physics.
    d, a, u = 5e-13, 2e-5, 1.5e-6
    assert ad.d_eff(d, a, u, "pipe") == pytest.approx(taylor_aris.d_eff(d, a, u), rel=1e-15)
    assert ad.velocity_variance(u, "pipe") == pytest.approx(
        taylor_aris.velocity_variance(u), rel=1e-15
    )

    # The channel is NOT the pipe; if these ever coincide the geometry
    # argument is being ignored somewhere.
    assert ad.DISPERSION_FACTOR["channel"] != ad.DISPERSION_FACTOR["pipe"]
    assert ad.VELOCITY_VARIANCE_FACTOR["channel"] != ad.VELOCITY_VARIANCE_FACTOR["pipe"]

    # The channel's COUPLED mode is pi^2, not its slowest transverse
    # eigenvalue pi^2/4; the latter would double the apparent time to reach
    # the Taylor regime.
    assert ad.COUPLED_EIGENVALUE["channel"] == pytest.approx(np.pi**2, rel=1e-15)
    assert ad.asymptotic_onset("channel") > ad.asymptotic_onset("pipe")

    # Green's function: exact moments at all times, not just asymptotically.
    t = 12.0
    x = np.linspace(-0.2, 0.4, 400001)
    c = ad.pulse_concentration(x, t, 1e-3, 4e-9)
    m0 = float(np.trapezoid(c, x))
    assert m0 == pytest.approx(1.0, rel=1e-9)
    assert float(np.trapezoid(x * c, x)) / m0 == pytest.approx(1e-3 * t, rel=1e-9)

    # Ballistic and asymptotic limits are DIFFERENT powers of t.
    for geom in ad.GEOMETRIES:
        early = ad.variance_short_time(np.array([1e-4, 2e-4]), 1.0, 1e-9, geom)
        assert early[1] / early[0] > 3.0, "short-time growth must be superlinear"


def test_advection_diffusion_rejects_unknown_geometry():
    """The geometry argument is validated, not silently defaulted."""
    with pytest.raises(ValueError):
        advection_diffusion.d_eff(1e-9, 1e-3, 1e-3, "annulus")


def test_advection_diffusion_release_symmetry_changes_the_fit_window():
    """The onset depends on the RELEASE, not on the geometry alone.

    A CORRECTION to the first version of this oracle, which claimed the pipe
    had no odd/even selection rule. It has the same rule and a larger gap: the
    Neumann Laplacian on a disc carries non-axisymmetric modes J_p(beta r/a)
    with J_p'(beta) = 0, and j'_{1,1}^2 = 3.390 is 4.331x slower than the
    j_{1,1}^2 = 14.682 the symmetric case uses. Nothing excites those modes
    while both u' and the release are axisymmetric — which is exactly why the
    original justification looked sound.
    """
    ad = advection_diffusion
    from scipy.special import jnp_zeros

    # The claimed slowest mode really is the slowest, and really is p != 0.
    slowest = min(b**2 for p in range(4) for b in jnp_zeros(p, 2) if b > 1e-9)
    assert slowest == pytest.approx(ad.ASYMMETRIC_EIGENVALUE["pipe"], rel=1e-6)
    assert slowest < ad.COUPLED_EIGENVALUE["pipe"], (
        "if the axisymmetric mode were the slowest, the original claim would "
        "have been right and this test is stale"
    )

    # Breaking the symmetry moves the fit window by more than 4x in both
    # geometries. A window chosen on the symmetric assumption is then wrong,
    # and silently so, because the fit still returns a number.
    for geom, factor in (("pipe", 4.331), ("channel", 4.0)):
        ratio = ad.asymptotic_onset(geom, symmetric_release=False) / ad.asymptotic_onset(geom)
        assert ratio == pytest.approx(factor, rel=1e-3)


def test_advection_diffusion_independent_anchors():
    """The anchors that are NOT the dispersion coefficient.

    Each is here because it catches something D_eff cannot.
    """
    ad = advection_diffusion

    # The third cumulant is blind to axial diffusion, and its SIGN differs
    # between the geometries — the only quantity here that does.
    assert ad.SKEWNESS_FACTOR["pipe"] > 0 > ad.SKEWNESS_FACTOR["channel"]

    # <u'^3> is an exact ZERO for a pipe. Zeros are much harder to hit by
    # accident than numbers.
    assert ad.THIRD_MOMENT_FACTOR["pipe"] == 0.0
    assert ad.THIRD_MOMENT_FACTOR["channel"] != 0.0

    # The variance intercept is negative and is NOT absorbable into a fitted
    # D_eff: it weights the same spectrum as beta^-8 rather than beta^-6.
    d, length, u = 2e-9, 1e-3, 5e-4
    for geom in ad.GEOMETRIES:
        assert ad.variance_intercept(length, u, d, geom) < 0.0

    # Released exactly on the u = U streamline, the pulse still ends up
    # permanently BEHIND. "Seeded at the mean velocity, so no offset" is the
    # plausible wrong answer this anchor exists to refute.
    for geom, xi0 in (("pipe", 1 / np.sqrt(2)), ("channel", 1 / np.sqrt(3))):
        assert ad.velocity_deviation(xi0, geom) == pytest.approx(0.0, abs=1e-15)
        assert ad.centroid_offset(xi0, length, u, d, geom) < 0.0

    # A cross-sectionally uniform release has NO centroid transient at all.
    assert ad.pulse_centroid(3.0, u) == pytest.approx(3.0 * u, rel=1e-15)

    # The transverse gate is a probability density in both geometries.
    for geom in ad.GEOMETRIES:
        lo = 0.0 if geom == "pipe" else -1.0
        xi = np.linspace(lo, 1.0, 200001)
        assert float(np.trapezoid(ad.transverse_pdf(xi, geom), xi)) == pytest.approx(
            1.0, rel=1e-12
        )

    # Balance Peclet: the two terms of D_eff are equal there, by construction.
    for geom in ad.GEOMETRIES:
        uu = ad.balance_peclet(geom) * d / length
        assert ad.d_eff(d, length, uu, geom) == pytest.approx(2 * d, rel=1e-12)
    assert ad.balance_peclet("pipe") == pytest.approx(np.sqrt(48), rel=1e-15)


def test_numerical_diffusion_self_verification():
    """The scheme's own error, as an oracle.

    Verified against a real 1-D solver before being committed: with the
    physical diffusivity set to exactly zero, the measured variance growth
    matches (u dx/2)(1 - Co) to a ratio of 1.0000 at nine combinations of mesh
    and Courant number, and 1.000000 up to N = 3200.
    """
    nd = numerical_diffusion
    out = nd.verify_limits()
    assert out, "verify_limits returned no checks"

    u, dx = 1.0, 1.0 / 400.0

    # The two structural anchors, which are sharper than the coefficient.
    assert nd.numerical_diffusivity(u, dx, 1.0) == 0.0, (
        "upwind at unit Courant number is a pure one-cell shift and therefore "
        "EXACT; measured variance change was -1.9e-17"
    )
    assert nd.numerical_diffusivity(u, dx, 0.0) == pytest.approx(0.5 * u * dx)

    # Monotone in Co: the temporal truncation cancels part of the spatial one.
    vals = [nd.numerical_diffusivity(u, dx, c) for c in (0.1, 0.5, 0.9)]
    assert vals[0] > vals[1] > vals[2] > 0.0

    # Central differencing is NEGATIVE at every Co. That is the instability,
    # written as a transport coefficient rather than a stability condition.
    for co in (0.05, 0.5, 0.95):
        assert nd.numerical_diffusivity(u, dx, co, "central_explicit") < 0.0

    # Dispersion-free Courant numbers are roots of E3, and E3 is not
    # identically zero, or the roots would be vacuous.
    for co in nd.dispersion_free_courant():
        assert nd.dispersive_coefficient(u, dx, co) == pytest.approx(0.0, abs=1e-18)
    assert nd.dispersive_coefficient(u, dx, 0.75) != 0.0

    # The point of the module: what a profile comparison actually measures is
    # D + D_num, and at haemodynamic parameters that is dominated by D_num.
    d_phys = 1e-9
    frac = nd.artefact_fraction(d_phys, 0.3, 5e-4, 0.5)
    assert frac > 0.99, (
        f"at u = 0.3 m/s, dx = 0.5 mm and D = 1e-9 m2/s the scheme contributes "
        f"{frac:.5f} of the spreading; if this ever drops below 0.99 the case "
        f"parameters have changed and the claim in the docstring is stale"
    )
    assert nd.total_spreading(d_phys, 0.3, 5e-4, 0.5) > 1e4 * d_phys


def test_numerical_diffusion_rejects_unknown_scheme():
    with pytest.raises(ValueError):
        numerical_diffusion.numerical_diffusivity(1.0, 0.01, 0.5, "quick")
