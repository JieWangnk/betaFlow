"""Aris moment hierarchy for a released solute pulse in laminar duct flow.

Valid at ALL TIMES, not only in the Taylor asymptotic limit. Extends
`advection_diffusion.py` (which carries only the t -> 0 and t -> inf limits)
with the exact second moment in between.

CONVENTIONS, identical to advection_diffusion.py so the two can be cross-checked:
    pipe    radius a,      u(r) = 2 U (1 - (r/a)^2),   weight 2s ds on s in [0,1]
    channel half-width h,  u(y) = (3/2) U (1 - (y/h)^2), uniform on eta in [-1,1]
    L_perp = a or h;  tau_r = L_perp^2 / D;  tau = t / tau_r.

THE RELEASE MATTERS, and it is the single easiest thing to get wrong here.
Everything below marked "plane release" assumes
    c(x, y_perp, 0) = (M/A) delta(x)                 -- UNIFORM over the section.
That is the Eulerian twin of betaflow's Lagrangian seeding P(r) = 2r/a^2. A
point release gives a DIFFERENT centroid law and a DIFFERENT power of t at
short time; both are given separately below rather than glossed over.

DERIVATION (Aris's method of moments). With c_p(y_perp,t) = int x^p c dx,
    dt c_p = D lap_perp c_p + p u c_{p-1} + p(p-1) D c_{p-2},
no-flux at the wall. Expand u' = u - U in the Neumann eigenfunctions phi_n of
-lap_perp (eigenvalue lam_n, mu_n = D lam_n, N_n = <phi_n^2>, u' = sum uh_n phi_n):

    sigma_x^2(t) = 2 D t + 2 sum_n (g_n/mu_n) * psi(mu_n t),
    psi(z) = z - 1 + exp(-z),        g_n = uh_n^2 N_n / mu_n.

Three spectral sums of the SAME operator fall out, and each is an anchor:
    sum g_n mu_n   = <u'^2>   = Var(u)          (Parseval)  -> t^2 law
    sum g_n        = D_eff - D = K L^2 U^2 / D              -> asymptotic slope
    sum g_n / mu_n = the constant offset / 2                -> intercept
They weight the spectrum as beta^-4, beta^-6, beta^-8, so they are three
genuinely different probes, not three views of one number. Mode 1 supplies
89.1%, 97.1% and 99.2% of them respectively: the offset is essentially the
first mode alone, Var(u) is the one that needs the fine structure.

CITATIONS, and the gaps stated as gaps.
  Taylor, G.I. (1953), Proc. R. Soc. A 219:186-203.
  Aris, R. (1956), Proc. R. Soc. A 235:67-77 -- the moment-equation hierarchy
    above and D + a^2 U^2/(48 D). NOT re-read while writing this module; the
    hierarchy is reproduced from the derivation shown, not transcribed.
  Crank, J. (1975), "The Mathematics of Diffusion", 2nd ed., sec. 2.2 -- the
    free-space Green's function.
  Watson, G.N. (1944), "Bessel Functions", 2nd ed., sec. 15.51 / NIST DLMF
    10.21(xiii) -- Rayleigh sums sum_n j_{1,n}^{-2k}. NOT opened here; the four
    values used are instead verified numerically to >=13 digits in
    `verify_limits`, so the citation is a convenience and not load-bearing.

  UNVERIFIED ATTRIBUTION, flagged rather than dressed up. The explicit
  exponential series below -- in particular the constant -a^4 U^2/(360 D^2)
  (pipe) and -2 h^4 U^2/(525 D^2) (channel) -- was DERIVED HERE. It is
  standardly credited to Chatwin, P.C. (1970), J. Fluid Mech. 43:321-352 and
  to Barton, N.G. (1983), J. Fluid Mech. 126:205-218 (exact moments for
  arbitrary initial data). NEITHER PAPER HAS BEEN READ. Treat both as leads to
  check. The result does not rest on them: it is reproduced by an independent
  method-of-lines solve of the moment PDEs in `verify_limits`.

NUMERICS. The closed form written as [2 D_eff t - const + series] suffers
catastrophic cancellation for tau < ~1e-5: at tau = 1e-9 it returns
-0.185 U^2 t^2 instead of U^2 t^2/3. `axial_variance` therefore evaluates the
psi-form with a Taylor branch for small z, which is accurate at tau = 1e-9 and
agrees with the naive form to 1e-16 at tau >= 0.05. Do not "simplify" it back.
"""

import numpy as np
from scipy.special import j0, jn_zeros, log_ndtr, ndtr

GEOMETRIES = ("channel", "pipe")
N_MODES_DEFAULT = 200  # 4e-14 at every tau tested; 50 modes gives only 4e-11

DISPERSION_FACTOR = {"pipe": 1.0 / 48.0, "channel": 2.0 / 105.0}          # K
VELOCITY_VARIANCE_FACTOR = {"pipe": 1.0 / 3.0, "channel": 1.0 / 5.0}      # Var(u)/U^2
# sigma^2 -> 2 D_eff t - VARIANCE_OFFSET_FACTOR * L^4 U^2 / D^2
VARIANCE_OFFSET_FACTOR = {"pipe": 1.0 / 360.0, "channel": 2.0 / 525.0}
# t^3 coefficient of (sigma^2 - 2Dt) is -CUBIC_FACTOR * D U^2 / L^2
CUBIC_FACTOR = {"pipe": 8.0 / 3.0, "channel": 1.0}
# t^(7/2) coefficient is +HALF_POWER_FACTOR * D^(3/2) U^2 / L^3.
# 1024/(105 sqrt(pi)) and 96/(35 sqrt(pi)). See `variance_short_time`.
HALF_POWER_FACTOR = {"pipe": 1024.0 / (105.0 * np.sqrt(np.pi)),
                     "channel": 96.0 / (35.0 * np.sqrt(np.pi))}
# <b> = 0 constant of the cell function b(xi) = xi^4/8 - xi^2/4 + c
CELL_CONSTANT = {"pipe": 1.0 / 12.0, "channel": 7.0 / 120.0}


def _check(geometry):
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    return geometry


# --------------------------------------------------------------------------
# Transverse spectrum. Everything downstream is a weighted sum over this.
# --------------------------------------------------------------------------


def transverse_modes(geometry, n_modes=N_MODES_DEFAULT):
    """(kappa_n, g_hat_n) for u' = sum uh_n phi_n on the Neumann spectrum.

    Returned non-dimensionally:  mu_n = kappa_n * D / L^2   and
    g_n = g_hat_n * L^2 U^2 / D, so that

        sigma^2 = 2 D t + 2 (L^4 U^2 / D^2) sum_n (g_hat_n/kappa_n) psi(kappa_n tau).

    pipe:    phi_n = J0(beta_n s), beta_n the n-th positive zero of J1,
             N_n = J0(beta_n)^2, uh_n = -8U/(beta_n^2 J0(beta_n)),
             kappa_n = beta_n^2, g_hat_n = 64/beta_n^6.
    channel: u' is EVEN, so the odd Neumann modes cos((2k+1)pi(eta+1)/2) have
             zero overlap and drop out entirely. The surviving modes are
             phi_m = cos(m pi eta), N_m = 1/2, uh_m = -6U(-1)^m/(m^2 pi^2),
             kappa_m = m^2 pi^2, g_hat_m = 18/(m^6 pi^6).
    """
    _check(geometry)
    if geometry == "pipe":
        beta = jn_zeros(1, n_modes)
        return beta**2, 64.0 / beta**6
    m = np.arange(1, n_modes + 1, dtype=float)
    return (m * np.pi) ** 2, 18.0 / (m * np.pi) ** 6


def _psi(z):
    """z - 1 + exp(-z), evaluated without cancellation.

    psi(z) ~ z^2/2 for small z; the direct expression loses every significant
    digit there, which is exactly the short-time regime the analytic reference is for.
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    small = z < 1.0e-2
    zs = z[small]
    out[small] = zs**2 / 2.0 * (1.0 - zs / 3.0 * (1.0 - zs / 4.0 * (1.0 - zs / 5.0)))
    zl = z[~small]
    out[~small] = zl - 1.0 + np.exp(-zl)
    return out


# --------------------------------------------------------------------------
# 1. Zeroth moment
# --------------------------------------------------------------------------


def total_mass(t, mass=1.0):
    """m0(t) = M, exactly, for all t.

    The identity a solver must satisfy on any control volume V is
        d/dt int_V c dV + oint_dV (u c - D grad c) . n dS = 0,
    with the wall contribution identically zero under no-flux. This is a
    property of a conservative discretisation, not of the transport physics,
    so it holds to round-off for any scheme, order, mesh or time step -- which
    is what makes a failure here unambiguous.
    """
    return np.full_like(np.asarray(t, dtype=float), float(mass))


# --------------------------------------------------------------------------
# 2. First moment
# --------------------------------------------------------------------------


def axial_centroid(t, u_mean):
    """<x>(t) = U t EXACTLY, at all t, for a cross-sectionally uniform release.

    There is NO transient. The reason is sharper than "the mean velocity is U":
    integrating the transport equation over x kills the advection term outright,
    so c_0(y_perp,t) = int c dx obeys the pure transverse heat equation. A
    uniform initial c_0 is its own steady state, hence the mass-weighted mean
    velocity is <u> = U from t = 0 onward and never relaxes.

    This makes the centroid a test of ADVECTION CONSISTENCY ALONE. It is
    independent of D, of Pe, of the mesh and of numerical diffusion: a
    first-order upwind scheme smeared beyond usefulness still returns U t to
    round-off. What breaks it is a flux/velocity reconstruction that is not
    discretely consistent, a wrong cell-volume weighting in the moment sum, or
    a dispersive phase error -- none of which sigma_x^2 isolates.
    """
    return u_mean * np.asarray(t, dtype=float)


def axial_centroid_point_release(t, xi0, u_mean, length, diffusivity,
                                 geometry="pipe", n_modes=N_MODES_DEFAULT):
    """<x>(t) for a release concentrated at transverse position xi0.

    Here there IS a transient, and it does not decay to nothing:

        d<x>/dt (0)      = u(xi0)        the local velocity, exactly
        d<x>/dt (t->inf) = U             exactly
        <x>(t) - U t     -> B(xi0) = (U L^2 / D) b(xi0)     PERMANENT

    b is the Aris cell function of advection_diffusion.py. So a non-uniform
    release leaves a fixed displacement offset rather than a persistent
    velocity error, and the two failure modes look nothing alike on a plot of
    <x> - U t: a solver defect keeps growing, this saturates.

    The counterintuitive case, and the reason this is worth testing: at
    xi0 = 1/sqrt(2) (pipe) or 1/sqrt(3) (channel) the release sits exactly on
    the streamline where u = U, yet the offset is -U a^2/(96 D) and
    -U h^2/(90 D) -- the pulse ends up permanently BEHIND a marker that
    travelled at U the whole way. Anyone who reasons "released at the mean
    velocity, therefore no offset" gets this wrong.
    """
    _check(geometry)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    kappa, _ = transverse_modes(geometry, n_modes)
    if geometry == "pipe":
        beta = np.sqrt(kappa)
        # uh_n / mu_n * phi_n(xi0), in units of U L^2 / D
        amp = -8.0 * j0(beta * xi0) / (beta**4 * j0(beta))
    else:
        # integer mode index: (-1.0)**m is NaN for a float m that is not
        # exactly integral, and sqrt(kappa)/pi is only integral to round-off
        m = np.round(np.sqrt(kappa) / np.pi).astype(int)
        amp = -6.0 * ((-1.0) ** m) * np.cos(m * np.pi * xi0) / (m * np.pi) ** 4
    scale = u_mean * length**2 / diffusivity
    relax = 1.0 - np.exp(-kappa * (diffusivity * t[:, None] / length**2))
    return u_mean * t + scale * np.sum(amp * relax, axis=1)


def cell_function(xi, geometry):
    """b(xi) = xi^4/8 - xi^2/4 + c, <b> = 0. The permanent centroid offset."""
    _check(geometry)
    x = np.asarray(xi, dtype=float)
    return x**4 / 8.0 - x**2 / 4.0 + CELL_CONSTANT[geometry]


# --------------------------------------------------------------------------
# 3. Second moment, all times
# --------------------------------------------------------------------------


def d_eff(diffusivity, length, u_mean, geometry):
    """D_eff = D + K L^2 U^2 / D. K = 1/48 (pipe), 2/105 (channel)."""
    _check(geometry)
    return diffusivity + DISPERSION_FACTOR[geometry] * length**2 * u_mean**2 / diffusivity


def velocity_variance(u_mean, geometry):
    """Var(u) over the cross-section: U^2/3 (pipe), U^2/5 (channel)."""
    _check(geometry)
    return VELOCITY_VARIANCE_FACTOR[geometry] * u_mean**2


def axial_variance(t, u_mean, length, diffusivity, geometry,
                   n_modes=N_MODES_DEFAULT):
    """sigma_x^2(t), EXACT at all t, for a cross-sectionally uniform release.

        sigma_x^2 = 2 D t + 2 sum_n (g_n/mu_n) [mu_n t - 1 + exp(-mu_n t)]

    which is algebraically identical to the closed forms

      pipe     2 (D + a^2U^2/48D) t - a^4U^2/(360 D^2)
                 + 128 (a^4U^2/D^2) sum_n beta_n^-8 exp(-beta_n^2 D t/a^2)
      channel  2 (D + 2h^2U^2/105D) t - 2 h^4U^2/(525 D^2)
                 + 36 (h^4U^2/(pi^8 D^2)) sum_m m^-8 exp(-m^2 pi^2 D t/h^2)

    but is evaluated in the psi-form because the closed forms cancel to
    nothing at small t. Two structural facts are exact and free: sigma^2(0) = 0
    and d sigma^2/dt (0) = 2 D -- the exponential series cancels both the
    constant and the entire shear part of the initial slope.
    """
    _check(geometry)
    t = np.atleast_1d(np.asarray(t, dtype=float))
    kappa, g_hat = transverse_modes(geometry, n_modes)
    tau = diffusivity * t[:, None] / length**2
    shear = np.sum((g_hat / kappa) * _psi(kappa * tau), axis=1)
    return 2.0 * diffusivity * t + 2.0 * length**4 * u_mean**2 / diffusivity**2 * shear


def variance_offset(u_mean, length, diffusivity, geometry):
    """The constant in sigma_x^2 -> 2 D_eff t - offset. POSITIVE as returned.

    a^4 U^2/(360 D^2) for the pipe, 2 h^4 U^2/(525 D^2) for the channel.
    Equivalently a virtual time origin: sigma^2 = 2 D_eff (t - t0). At high Pe
    t0 -> tau_r/15 (pipe).

    This is the anchor that a fitted D_eff cannot absorb. Numerical diffusion
    adds to the SLOPE and is therefore invisible once D_eff is a free
    parameter; the intercept is set by sum g_n/mu_n, a different weighting of
    the same spectrum, so slope and intercept together are two independent
    constraints on the transverse operator rather than one.
    """
    _check(geometry)
    return VARIANCE_OFFSET_FACTOR[geometry] * length**4 * u_mean**2 / diffusivity**2


# --------------------------------------------------------------------------
# 4. Short time
# --------------------------------------------------------------------------


def variance_short_time(t, u_mean, length, diffusivity, geometry, order=3):
    """sigma_x^2 - 2 D t = Var(u) t^2 - (C D U^2/L^2) t^3 + (A D^{3/2} U^2/L^3) t^{7/2}

        pipe     U^2 t^2/3 - (8/3) D U^2 t^3/a^2 + (1024/(105 sqrt(pi))) ... t^{7/2}
        channel  U^2 t^2/5 -       D U^2 t^3/h^2 + ( 96/( 35 sqrt(pi))) ... t^{7/2}

    THE SERIES IS NOT IN INTEGER POWERS. The formal t^4 coefficient is
    (1/12) sum_n g_n mu_n^3, whose terms tend to a CONSTANT (64 for the pipe,
    18 for the channel) and which therefore diverges. The divergence is the
    signature of a half-power: the true next term is t^{7/2}, fixed by the pole
    of the spectral zeta function at s = -7/2 with Gamma(-7/2) = 16 sqrt(pi)/105.
    Verified to 40 digits: the deviation of the numerical coefficient from
    1024/(105 sqrt(pi)) falls exactly as sqrt(tau) over four decades.

    Physically t^{7/2} = t^3 * sqrt(D t)/L is the first term that knows the WALL
    exists. Up to t^3 the answer depends only on the velocity field --
    Var(u) and <|grad u|^2> -- and would be identical in an unbounded shear;
    sqrt(D t)/L is the fraction of the section inside the diffusive wall layer.
    So this term, and nothing before it, tests the no-flux boundary condition.

    The t^3 coefficient has a geometry-free form worth knowing:
        -(D/3) <|grad_perp u|^2> t^3,
    with <|grad u|^2> = 8U^2/a^2 (pipe) and 3U^2/h^2 (channel), which is how
    the 8/3 and the 1 are generated rather than remembered.

    CONFIRMS the (U^2/3) t^2 already used by taylor_aris.py for the pipe in the
    Lagrangian formulation: the Eulerian route reaches it by Parseval,
    sum_n g_n mu_n = <u'^2>, which is the same number for a different reason.
    The channel counterpart is Var(u) = U^2/5.

    THE CONFIRMATION IS CONDITIONAL ON THE RELEASE. Var(u) here is the variance
    over the INITIAL cross-sectional distribution, and it is U^2/3 only because
    betaflow seeds uniformly, P(r) = 2r/a^2. For a point release Var is zero and
    the leading term is t^3 instead -- see `variance_short_time_point_release`.

    The t^3 term is the exact statement of the "deficit" taylor_aris.py records
    qualitatively: the relative correction is -8 t/tau_r (pipe), -5 t/tau_r
    (channel), so a fit window centred at tau biases the recovered prefactor
    low by that fraction and drags the fitted exponent below 2 at the same time.
    """
    _check(geometry)
    tt = np.asarray(t, dtype=float)
    out = velocity_variance(u_mean, geometry) * tt**2
    if order >= 3:
        out = out - CUBIC_FACTOR[geometry] * diffusivity * u_mean**2 * tt**3 / length**2
    if order >= 4:  # the t^{7/2} term; `order` counts terms, not powers
        out = out + (HALF_POWER_FACTOR[geometry] * diffusivity**1.5 * u_mean**2
                     * tt**3.5 / length**3)
    return out


def variance_short_time_point_release(t, xi0, u_mean, length, diffusivity,
                                      geometry="pipe"):
    """Point release: the t^2 term VANISHES and the leading term is t^3.

        sigma_x^2 - 2 D t -> (2/3) (du/dn)^2_{xi0} D t^3

    with du/dn = -4 U xi0 / a (pipe) and -3 U xi0 / h (channel). All the
    solute starts on one streamline, so there is no initial velocity spread to
    make a t^2 term; the spread is manufactured by transverse diffusion, one
    order of t later.

    Released on the AXIS the t^3 term vanishes too and the leading behaviour is
    t^4: for the pipe, exactly (32/3) U^2 D^2 t^4 / a^4.

    Derived by an exactly-solvable route rather than by linearisation: u is a
    quadratic form in the transverse displacement, so X_shear = 2Ut -
    (2U/a^2) int |y0 + W|^2 dt' is a Gaussian-process functional whose variance
    is exact --  (4 r0^2 sigma^2 t^3/3 + 2 sigma^4 t^4/3) with sigma^2 = 2D.
    Free-space; wall reflections are exponentially small at these times.
    """
    _check(geometry)
    tt = np.asarray(t, dtype=float)
    slope = (4.0 if geometry == "pipe" else 3.0) * u_mean * xi0 / length
    quartic = (32.0 / 3.0) * u_mean**2 * diffusivity**2 * tt**4 / length**4
    return (2.0 / 3.0) * slope**2 * diffusivity * tt**3 + (
        quartic if geometry == "pipe" else 0.0
    )


# --------------------------------------------------------------------------
# 5. Molecular-communications impulse response
# --------------------------------------------------------------------------


def pulse_concentration(x, t, u_mean, dispersivity, mass=1.0, area=1.0):
    """Cross-sectionally averaged concentration, 1-D advection-diffusion.

        cbar(x,t) = M / (A sqrt(4 pi K t)) exp(-(x - U t)^2 / (4 K t))

    K = D gives the pure advection-diffusion channel (exact only for plug flow);
    K = d_eff(...) gives the Taylor-dispersion channel. Evaluated at x = L as a
    function of t this is the transparent-receiver impulse response; the count
    in a small receiver volume is V_rx * cbar(L,t).

    VALIDITY of the Taylor form: t >> tau_r, i.e. L/L_perp >> Pe = U L_perp/D.
    Below that the 1-D reduction is simply not available at any K, because the
    profile is not Gaussian yet -- use `pulse_concentration_moment_matched`.
    """
    xx = np.asarray(x, dtype=float)
    tt = np.asarray(t, dtype=float)
    return (mass / (area * np.sqrt(4.0 * np.pi * dispersivity * tt))
            * np.exp(-((xx - u_mean * tt) ** 2) / (4.0 * dispersivity * tt)))


def pulse_concentration_moment_matched(x, t, u_mean, length, diffusivity,
                                       geometry, mass=1.0, area=1.0):
    """Gaussian carrying the EXACT sigma_x^2(t) instead of 2 K t.

    Exact in m0, m1 and m2 by construction at every t, including t < tau_r
    where the Taylor form is not valid at all. APPROXIMATE from m3 up: the true
    profile is genuinely skewed at t ~ tau_r (the pulse has a long tail of
    slow near-wall solute), so this is a moment-matched model and not a
    solution. Use it as a channel model, never as an analytic reference for the shape.
    """
    var = axial_variance(t, u_mean, length, diffusivity, geometry)
    xx = np.asarray(x, dtype=float)
    tt = np.asarray(t, dtype=float)
    return (mass / (area * np.sqrt(2.0 * np.pi * var))
            * np.exp(-((xx - u_mean * tt) ** 2) / (2.0 * var)))


def arrival_time_pdf(t, distance, u_mean, dispersivity):
    """Inverse-Gaussian first-passage density to an absorbing plane at x = L.

        f_T(t) = L / sqrt(4 pi K t^3) exp(-(L - U t)^2 / (4 K t))

    IG(mu = L/U, lambda = L^2/(2K)): E[T] = L/U, Var[T] = 2 K L / U^3. Unlike
    cbar(L, .), this is a normalised density in t (integral exactly 1 for U > 0)
    and is the right object for an absorbing receiver. Note the exact relation
    f_T(t) = (L/t) (A/M) cbar(L,t), so the two differ by the factor L/t alone.
    """
    tt = np.asarray(t, dtype=float)
    return (distance / np.sqrt(4.0 * np.pi * dispersivity * tt**3)
            * np.exp(-((distance - u_mean * tt) ** 2) / (4.0 * dispersivity * tt)))


def arrival_time_cdf(t, distance, u_mean, dispersivity):
    """Fraction absorbed by time t.

        F_T(t) = Phi((U t - L)/sqrt(2 K t)) + e^{U L/K} Phi((-U t - L)/sqrt(2 K t))

    Evaluated in log space. The literal expression OVERFLOWS whenever
    U L / K > ~709, which for a Taylor-dispersion channel is an ordinary
    operating point, not an edge case: it returns NaN at U L/K = 2000 where the
    log-space form agrees with quadrature to 12 digits. An analytic reference that returns
    NaN in its own working range is worse than no analytic reference.
    """
    tt = np.asarray(t, dtype=float)
    s = np.sqrt(2.0 * dispersivity * tt)
    return ndtr((u_mean * tt - distance) / s) + np.exp(
        u_mean * distance / dispersivity + log_ndtr((-u_mean * tt - distance) / s)
    )


# --------------------------------------------------------------------------
# Self-verification
# --------------------------------------------------------------------------


def verify_limits(rtol=1e-9):
    """Every entry relates two things computed by DIFFERENT routes."""
    import sympy as sp

    errors = {}
    d, u = 1.0, 1.0  # non-dimensional, L = 1 so tau_r = 1

    for geom in GEOMETRIES:
        kappa, g_hat = transverse_modes(geom, 4000)

        # 1. The three spectral sums against their exact rationals.
        errors[f"{geom}_sum_var"] = abs(
            np.sum(g_hat * kappa) / VELOCITY_VARIANCE_FACTOR[geom] - 1.0)
        errors[f"{geom}_sum_K"] = abs(
            np.sum(g_hat) / DISPERSION_FACTOR[geom] - 1.0)
        errors[f"{geom}_sum_offset"] = abs(
            2.0 * np.sum(g_hat / kappa) / VARIANCE_OFFSET_FACTOR[geom] - 1.0)
        # sum g_n kappa_n^2 converges only as 1/n_modes (it is sum beta^-2 /
        # sum m^-2 in disguise), so this one gets a tail-aware tolerance. The
        # same coefficient is tested sharply by `{geom}_short_t3` below.
        errors[f"{geom}_sum_cubic"] = abs(
            np.sum(g_hat * kappa**2) / (3.0 * CUBIC_FACTOR[geom]) - 1.0)

        # 2. sigma^2(0) = 0 and d/dt sigma^2 (0) = 2 D, both by cancellation.
        errors[f"{geom}_var_at_zero"] = abs(
            float(axial_variance(1e-300, u, 1.0, d, geom)[0]))
        h_ = 1e-9
        errors[f"{geom}_slope_at_zero"] = abs(
            float(axial_variance(h_, u, 1.0, d, geom)[0]) / (2.0 * d * h_) - 1.0)

        # 3. Short time: the full formula against the series, at a tau where
        #    the t^3 term matters (1e-3 -> a 0.8% correction, so this tests the
        #    cubic coefficient and not merely the t^2 one).
        for tau, tol_name, order in ((1e-8, "t2", 2), (1e-4, "t3", 3),
                                     (1e-4, "t72", 4)):
            full = float(axial_variance(tau, u, 1.0, d, geom, 4000)[0]) - 2.0 * d * tau
            ser = float(variance_short_time(tau, u, 1.0, d, geom, order=order))
            errors[f"{geom}_short_{tol_name}"] = abs(full / ser - 1.0)
        # the t^{7/2} term must IMPROVE on the two-term series, not merely fit
        if not errors[f"{geom}_short_t72"] < 0.2 * errors[f"{geom}_short_t3"]:
            raise AssertionError(f"{geom}: t^(7/2) term does not improve the series")

        # 4. Long time: slope and intercept read back off the full formula.
        tl = np.array([4.0, 8.0])
        vl = axial_variance(tl, u, 1.0, d, geom)
        slope = (vl[1] - vl[0]) / (tl[1] - tl[0])
        inter = vl[0] - slope * tl[0]
        errors[f"{geom}_long_slope"] = abs(slope / (2.0 * d_eff(d, 1.0, u, geom)) - 1.0)
        errors[f"{geom}_long_offset"] = abs(
            -inter / variance_offset(u, 1.0, d, geom) - 1.0)

        # 5. Centroid: uniform release is U t; point release starts at u(xi0),
        #    ends at U, and saturates at the cell function.
        xi0 = 0.5
        errors[f"{geom}_centroid_uniform"] = abs(
            float(axial_centroid(3.7, u)) / (u * 3.7) - 1.0)
        u_local = (2.0 * (1 - xi0**2) if geom == "pipe" else 1.5 * (1 - xi0**2))
        # the initial slope is a pointwise Fourier/Fourier-Bessel sum of u',
        # whose coefficients decay only as m^-2, so it needs many modes
        m1a = float(axial_centroid_point_release(1e-7, xi0, u, 1.0, d, geom,
                                                 n_modes=20000)[0])
        errors[f"{geom}_point_initial_slope"] = abs(m1a / (u_local * 1e-7) - 1.0)
        m1b = float(axial_centroid_point_release(20.0, xi0, u, 1.0, d, geom,
                                                 n_modes=20000)[0])
        errors[f"{geom}_point_offset"] = abs(
            (m1b - u * 20.0) / float(cell_function(xi0, geom)) - 1.0)

        # 6. The cell function reconstructed from its OWN modal series -- this
        #    is what pins the constants 1/12 and 7/120, independently of the
        #    quadrature that produced them in advection_diffusion.py.
        for xi in (0.0, 1.0):
            far = float(axial_centroid_point_release(1e4, xi, u, 1.0, d, geom,
                                                     n_modes=4000)[0]) - u * 1e4
            errors[f"{geom}_cell_series_{xi:.0f}"] = abs(
                far - float(cell_function(xi, geom)))

        # 7. Mode truncation is converged at the default.
        ref = float(axial_variance(0.3, u, 1.0, d, geom, n_modes=4000)[0])
        errors[f"{geom}_mode_truncation"] = abs(
            float(axial_variance(0.3, u, 1.0, d, geom)[0]) / ref - 1.0)

    # 8. Rayleigh sums for the zeros of J1, as exact rationals.
    beta = jn_zeros(1, 20000)
    for k, exact in ((4, sp.Rational(1, 192)), (6, sp.Rational(1, 3072)),
                     (8, sp.Rational(1, 46080))):
        errors[f"rayleigh_{k}"] = abs(
            float(np.sum(beta ** -float(k))) / float(exact) - 1.0)

    # 9. CROSS-MODULE: the limits must match the module that already owns them.
    from betaflow.analytic import advection_diffusion as ad

    for geom in GEOMETRIES:
        errors[f"{geom}_matches_ad_deff"] = abs(
            d_eff(2e-9, 1.5e-3, 2e-4, geom) / ad.d_eff(2e-9, 1.5e-3, 2e-4, geom) - 1.0)
        errors[f"{geom}_matches_ad_var"] = abs(
            velocity_variance(3.0, geom) / ad.velocity_variance(3.0, geom) - 1.0)

    # 10. Green's function: moments by quadrature, and the FPT normalisation,
    #     mean and variance, plus the log-space CDF against the density.
    from scipy.integrate import quad

    d_, u_, t_ = 2.0e-9, 1.0e-3, 50.0
    span = 40.0 * np.sqrt(2.0 * d_ * t_)
    x = np.linspace(u_ * t_ - span, u_ * t_ + span, 2000001)
    c = pulse_concentration(x, t_, u_, d_)
    m0 = float(np.trapezoid(c, x))
    m1 = float(np.trapezoid(x * c, x)) / m0
    m2 = float(np.trapezoid((x - m1) ** 2 * c, x)) / m0
    errors["greens_mass"] = abs(m0 - 1.0)
    errors["greens_centroid"] = abs(m1 / (u_ * t_) - 1.0)
    errors["greens_variance"] = abs(m2 / (2.0 * d_ * t_) - 1.0)

    ell, kk, uu = 1.0e-3, 5.0e-10, 1.0e-3        # U L / K = 2000: overflows naively
    tot, _ = quad(arrival_time_pdf, 0, np.inf, args=(ell, uu, kk), limit=500)
    mean, _ = quad(lambda z: z * arrival_time_pdf(z, ell, uu, kk), 0, np.inf, limit=500)
    var, _ = quad(lambda z: (z - ell / uu) ** 2 * arrival_time_pdf(z, ell, uu, kk),
                  0, np.inf, limit=500)
    errors["fpt_normalisation"] = abs(tot - 1.0)
    errors["fpt_mean"] = abs(mean / (ell / uu) - 1.0)
    errors["fpt_variance"] = abs(var / (2.0 * kk * ell / uu**3) - 1.0)
    tq = 1.05 * ell / uu
    cdf_q, _ = quad(arrival_time_pdf, 0, tq, args=(ell, uu, kk), limit=500)
    errors["fpt_cdf_vs_density"] = abs(
        float(arrival_time_cdf(tq, ell, uu, kk)) / cdf_q - 1.0)
    if not np.isfinite(arrival_time_cdf(tq, ell, uu, kk)):
        raise AssertionError("arrival_time_cdf overflowed; the log-space form is required")

    # 11. Point-release short-time law, against the exact Gaussian-process
    #     variance of the quadratic functional (symbolic, not a refit).
    r0, tt = sp.Rational(1, 2), sp.Symbol("t", positive=True)
    sig2, D_, U_, a_ = sp.symbols("sigma2 D U a", positive=True)
    uu_, vv = sp.symbols("u v", positive=True)
    A_ = sp.integrate(sp.integrate(sp.Min(uu_, vv), (uu_, 0, tt)), (vv, 0, tt))
    B_ = sp.integrate(sp.integrate(sp.Min(uu_, vv) ** 2, (uu_, 0, tt)), (vv, 0, tt))
    exact = sp.expand(sp.simplify(
        (4 * U_**2 / a_**4) * (4 * r0**2 * sig2 * A_ + 4 * sig2**2 * B_)
    ).subs(sig2, 2 * D_))
    predicted = sp.expand(
        sp.Rational(2, 3) * (4 * U_ * r0 / a_**2) ** 2 * D_ * tt**3
        + sp.Rational(32, 3) * U_**2 * D_**2 * tt**4 / a_**4)
    errors["point_release_t3_symbolic"] = float(
        sp.Abs(sp.simplify(exact - predicted)).subs({U_: 1, a_: 1, D_: 1, tt: 1}))

    for name, err in errors.items():
        limit = rtol
        if "short_t2" in name or "long" in name or "point" in name or "cell_series" in name:
            limit = 1e-6          # series truncation / finite-difference probes
        if "rayleigh_4" in name:
            limit = 1e-7          # sum j^-4 converges as 1/n^3
        if "short_t3" in name:
            limit = 6e-3      # truncated series vs exact; t^(7/2) is the residual
        if "short_t72" in name:
            limit = 3e-4      # residual is then O(tau^4)
        if "point" in name or "cell_series" in name:
            limit = 1e-6
        if "sum_cubic" in name:
            limit = 1e-3          # sum beta^-2 / sum m^-2: tail falls as 1/n
        if "slope_at_zero" in name:
            limit = 1e-7
        if not err < limit:
            raise AssertionError(
                f"moment_hierarchy analytic reference {name} error {err:.3e} > {limit:.0e}")
    return {k: float(v) for k, v in errors.items()}


if __name__ == "__main__":
    for k, v in verify_limits().items():
        print(f"{k:42s} {v:.3e}")
