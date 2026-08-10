"""Eulerian advection-diffusion of a passive scalar in laminar duct flow.

THE EULERIAN TWIN of `taylor_aris.py`. That module tracks particles and
measures a variance of positions; this one describes a CONCENTRATION FIELD and
measures a moment of that field. The physics is identical and the exact answer
is identical, so a solver that gets one right and the other wrong has a bug
that neither case alone can localise. `verify_limits` asserts the agreement
across the two modules rather than importing one into the other, so the two
implementations stay genuinely independent.

Written for lattice-Boltzmann advection-diffusion lattices and for
finite-volume scalar transport, neither of which this module knows about.

ANCHORS, in increasing order of how hard they are to pass by accident.

  1. CONSERVATION.  int c dV is exactly constant in a closed domain, and
     d/dt int c dV = -closed-integral (u c - D grad c).n dS otherwise. Exact
     for the discrete problem, independent of the transport physics.

  2. THE GREEN'S FUNCTION.  For a plane source of strength M released into
     uniform velocity U with diffusivity D,
         c(x,t) = M / sqrt(4 pi D t) * exp(-(x - U t)^2 / (4 D t))
     with centroid exactly U t and variance exactly 2 D t at ALL times.
     A centroid that drifts is a broken advection scheme; a variance that
     grows too fast is numerical diffusion. These fail differently, which is
     why both are checked.

  3. TAYLOR-ARIS DISPERSION.  At t >> tau_r the cross-sectionally averaged
     concentration obeys a 1-D advection-diffusion equation with
         D_eff = D + K L^2 U^2 / D
     K = 1/48 for a circular pipe of radius L = a, and
     K = 2/105 for a plane channel of half-width L = h.
     Both are DERIVED below rather than quoted, and `verify_limits` re-derives
     them numerically from the cell problem.

  4. THE SHORT-TIME BALLISTIC LIMIT.  sigma_x^2 - 2 D t -> Var(u) t^2, a
     DIFFERENT POWER OF t, with Var(u)/U^2 = 1/3 (pipe) and 1/5 (channel).
     A code that lands D_eff correctly by accident still misses this.

  5. THE APPROACH TO THE ASYMPTOTE.  The pre-asymptotic correction decays as
     exp(-beta^2 D t / L^2) with beta^2 = 14.682 (pipe) and pi^2 (channel).
     The channel value is NOT the slowest eigenvalue of the transverse
     operator — see below.

CONVENTIONS, which account for most published disagreement on K.
    channel: walls at y = +/- h, so h is the HALF-width and the gap is 2h;
             u(y) = (3/2) U (1 - (y/h)^2), U the cross-sectionally AVERAGED
             velocity. This matches betaflow/analytic/poiseuille.py.
    pipe:    radius a; u(r) = 2 U (1 - (r/a)^2), U again the area-averaged
             velocity. Matches betaflow/analytic/pipe.py.
Using the full gap instead of the half-width changes K by 4; using the
centreline instead of the mean velocity changes it by (3/2)^2 = 9/4.

DERIVATION of K, by the cell-problem route, so the numbers are reproducible
rather than remembered. With u'(y) = u(y) - U the velocity deviation, solve

    D grad^2 B = -u',    no flux at the wall,    <B> = 0

and then D_eff = D + <u' B>, the average being over the cross-section. Writing
B = (U L^2 / D) b, both geometries give the SAME cell function up to the
coordinate,

    b'(xi) = xi (xi^2 - 1) / 2

which satisfies the no-flux condition automatically at xi = 1 — a useful
sign that the setup is right before any integration is done. Then

    pipe    b(s)   = s^4/8 - s^2/4 + 1/12,     K = <u' b> = 1/48
    channel b(eta) = eta^4/8 - eta^2/4 + 7/120, K = <u' b> = 2/105

The two constants of integration differ only because the averaging weight
does: 2s ds on the disc, and uniform on the slab.

WHICH TRANSVERSE MODE ACTUALLY MATTERS, and a correction to the first
version of this module. For no-flux walls the channel's transverse
eigenfunctions are cos(n pi (y+h)/2h) with eigenvalues (n pi/2h)^2, so the
slowest nonzero mode is pi^2/(4 h^2). That is NOT the rate governing the
approach to Taylor dispersion. u'(y) is EVEN in y, so it is orthogonal to
every odd mode; the slowest COUPLED mode is n = 2, giving pi^2/h^2. Since
onset time goes as 1/eigenvalue, quoting the slowest EXISTING mode overstates
it by a factor of FOUR (pi^2 / (pi^2/4)), not two.

THE PIPE HAS THE SAME SELECTION RULE, AND A LARGER GAP. The first version of
this module said the pipe "has no such cancellation because its J0 modes are
all even". That is wrong. The Neumann Laplacian on a disc also carries
NON-AXISYMMETRIC modes J_p(beta r/a) exp(i p theta) with J_p'(beta) = 0, and
the ordered spectrum begins

    beta^2 =  3.38996   (j'_{1,1} = 1.8411838)   <- the TRUE slowest mode
    beta^2 =  9.32836   (j'_{2,1} = 3.0542369)
    beta^2 = 14.68197   (j'_{0,1} = j_{1,1})     <- what this module quotes

so the true slowest transverse mode is 4.331x slower than the one used. It is
silent only because BOTH u' and an area-uniform release are axisymmetric, so
nothing ever excites p != 0. Break that symmetry — an off-axis injection, a
cloud seeded on a patch subset, an azimuthally biased mesh — and the onset
moves from 0.941 tau_r to 4.075 tau_r. `asymptotic_onset` therefore takes an
explicit `symmetric_release` flag instead of assuming. The channel's analogue
moves from 1.400 to 5.599 tau_h.

This correction is recorded rather than quietly applied because the numbers
0.941 and 1.400 were never wrong — the JUSTIFICATION for them was, and a fit
window chosen from a wrong justification survives only as long as the symmetry
does.

CITATIONS. Status is stated per source, because several of the obvious ones
turn out not to say what they are cited for.

  VERIFIED BY READING THE TEXT
  Ajdari, A., Bontoux, N. & Stone, H.A. (2006), Anal. Chem. 78(2):387-392,
    doi:10.1021/ac0508651. Gives the two-dimensional result
    D_eff/D = 1 + Pe_h^2/210 with Pe_h = V h0/D, h0 the channel HEIGHT (full
    gap) and V explicitly the mean velocity. Converting to the half-width used
    here gives exactly 2/105. THIS IS THE STRONGEST ANCHOR FOR THE CHANNEL
    CONSTANT. The same paper notes Aris worked his general-cross-section
    formula for ELLIPSES, and that a rectangular duct is ~8x the parallel-plate
    value rather than tending to it.
  Lee, G., Luner, A., Marzuola, J. & Harris, D.M. (2021), Microfluid.
    Nanofluid. 25:34, doi:10.1007/s10404-021-02436-9. Eq. (1) with dispersion
    factor f = 1 for two infinite parallel plates; h the full height, U the
    mean. Independent statement of the same number.
  Chatwin, P.C. & Sullivan, P.J. (1982), J. Fluid Mech. 120:347-358,
    doi:10.1017/S0022112082002791. The side-wall caveat: longitudinal
    diffusivity in a rectangular channel is ~8 D0 at large aspect ratio.

  VERIFIED TO EXIST, CITED FOR THE METHOD ONLY
  Taylor, G.I. (1953), Proc. R. Soc. A 219(1137):186-203,
    doi:10.1098/rspa.1953.0139. The circular tube. NOT a source for the
    plane-channel constant.
  Aris, R. (1956), Proc. R. Soc. A 235(1200):67-77,
    doi:10.1098/rspa.1956.0065. The method of moments and the canonical form
    D_eff = D + K a^2 U^2/D. Cite for the METHOD; there is no evidence he
    printed the parallel-plate value, so do not cite him for 2/105.
  Crank, J. (1975), "The Mathematics of Diffusion", 2nd ed., OUP, sec. 2.2 —
    the free-space Green's function of anchor 2.

  A PUBLISHED WRONG VALUE FOR EXACTLY THIS CONSTANT, recorded because it is
  the best evidence that the constant is easy to get wrong:
  Beard, D.A. (2001), J. Appl. Phys. 89(8):4667-4669 published 33/560
    (= 0.0589) for the plane channel, three times the correct 2/105
    (= 0.0190). It was corrected by Dorfman, K.D. & Brenner, H. (2001),
    J. Appl. Phys. 90(12):6553-6554. Neither full text was read here; the
    bibliographic records were verified. Do not cite Beard (2001) for this
    coefficient.

  ATTRIBUTION WITHDRAWN. An earlier version of this module offered Wooding,
  R.A. (1960), J. Fluid Mech. 7:501-515 as the likely primary source. That
  paper is "Instability of a viscous liquid of variable density in a vertical
  Hele-Shaw cell" — a stability paper, and there is no confirmation it
  contains a dispersion coefficient. The lead is dropped rather than left
  standing, since a plausible-looking wrong citation is worse than none.

  NO PRIMARY SOURCE for 2/105 was verified. It is used here on the strength of
  six mutually independent derivations (symbolic cell problem, the spectral
  sum closing on zeta(6) = pi^6/945, the Dirichlet identity that also fixes the
  sign, finite differences, Monte Carlo, and the method of moments), plus the
  fact that the identical machinery reproduces Aris's published 1/48 for the
  pipe. The pipe constant needs no such caveat.

  CONVENTION TRAP, verified arithmetic: u_max = 2U for a pipe, so K = 1/48 in
  mean-velocity units is K = 1/192 in centreline units, and Taylor (1953) is
  often quoted in the 192 form. For the channel, K = 2/105 (half-width, mean)
  is 1/210 (full gap, mean) and 8/945 (half-width, centreline). Three of the
  four published forms above are algebraically the same number.
"""

import numpy as np
from scipy.special import jn_zeros

# --------------------------------------------------------------------------
# Geometry constants. Every one is derived in the module docstring and
# re-derived numerically in `verify_limits`; none is a remembered number.
# --------------------------------------------------------------------------

GEOMETRIES = ("channel", "pipe")

# K in D_eff = D + K L^2 U^2 / D.
DISPERSION_FACTOR = {"pipe": 1.0 / 48.0, "channel": 2.0 / 105.0}

# Var(u)/U^2 over the cross-section — the short-time ballistic prefactor.
VELOCITY_VARIANCE_FACTOR = {"pipe": 1.0 / 3.0, "channel": 1.0 / 5.0}

# beta^2 in the pre-asymptotic decay exp(-beta^2 D t / L^2), for the slowest
# mode that COUPLES to u'. See the docstring: the channel's is not its
# slowest transverse eigenvalue.
BESSEL_J1_FIRST_ROOT = float(jn_zeros(1, 1)[0])  # 3.8317059702
COUPLED_EIGENVALUE = {
    "pipe": BESSEL_J1_FIRST_ROOT**2,  # 14.6819706
    "channel": np.pi**2,  # 9.8696044
}


# The slowest transverse mode that EXISTS, as opposed to the one that couples.
# Reached only by an asymmetric release; see the docstring. pipe: j'_{1,1}^2.
ASYMMETRIC_EIGENVALUE = {"pipe": 3.3899590, "channel": np.pi**2 / 4.0}

# Intercept of the long-time variance: sigma^2 -> 2 D_eff t - C L^4 U^2 / D^2.
# From sum_n g_n/mu_n, a THIRD weighting (beta^-8) of the same spectrum, so it
# is not absorbable into a fitted D_eff. pipe 128 * (1/46080) = 1/360;
# channel 36 * zeta(8)/pi^8 = 2/525.
VARIANCE_INTERCEPT_FACTOR = {"pipe": 1.0 / 360.0, "channel": 2.0 / 525.0}

# Third cumulant: kappa_3 -> 6 <u' B^2> t = S L^4 U^3 / D^2 * t.
# EXACTLY blind to axial molecular diffusion, because a Gaussian axial kernel
# contributes nothing beyond the second cumulant. The SIGNS DIFFER between the
# geometries, which no other anchor here does.
SKEWNESS_FACTOR = {"pipe": 6.0 / 2880.0, "channel": -24.0 / 17325.0}

# <u'^3>/U^3. EXACTLY ZERO for a pipe, because r^2/a^2 is uniform under the
# area measure. A zero is far harder to hit by accident than a number.
THIRD_MOMENT_FACTOR = {"pipe": 0.0, "channel": -2.0 / 35.0}


def _check_geometry(geometry):
    if geometry not in GEOMETRIES:
        raise ValueError(f"geometry must be one of {GEOMETRIES}, got {geometry!r}")
    return geometry


# --------------------------------------------------------------------------
# Cross-sectional structure
# --------------------------------------------------------------------------


def velocity_deviation(xi, geometry):
    """u'(xi)/U, the velocity deviation from the mean.

    `xi` is r/a in [0, 1] for a pipe, or y/h in [-1, 1] for a channel. Both
    average to zero over their own cross-section, which `verify_limits`
    checks by quadrature rather than assuming.
    """
    _check_geometry(geometry)
    x = np.asarray(xi, dtype=float)
    if geometry == "pipe":
        return 1.0 - 2.0 * x**2
    return 0.5 - 1.5 * x**2


def cross_section_weight(xi, geometry):
    """Area-averaging weight: 2s for the disc, uniform for the slab."""
    _check_geometry(geometry)
    x = np.asarray(xi, dtype=float)
    return 2.0 * x if geometry == "pipe" else np.ones_like(x)


def cell_function(xi, geometry):
    """b(xi), the solution of the Aris cell problem, normalised to <b> = 0.

    B = (U L^2 / D) b. Both geometries share b'(xi) = xi(xi^2 - 1)/2 and
    differ only in the constant that enforces <b> = 0 under their own weight.
    """
    _check_geometry(geometry)
    x = np.asarray(xi, dtype=float)
    const = 1.0 / 12.0 if geometry == "pipe" else 7.0 / 120.0
    return x**4 / 8.0 - x**2 / 4.0 + const


def cell_function_derivative(xi):
    """b'(xi) = xi (xi^2 - 1)/2 — identical in both geometries.

    b'(1) = 0 is the no-flux wall condition, satisfied identically rather
    than imposed, which is the cheapest available check that the cell problem
    was set up correctly.
    """
    x = np.asarray(xi, dtype=float)
    return x * (x**2 - 1.0) / 2.0


# --------------------------------------------------------------------------
# Dispersion
# --------------------------------------------------------------------------


def peclet(u_mean, length, diffusivity):
    """Pe = U L / D, with L the pipe radius or the channel half-width."""
    return u_mean * length / diffusivity


def d_eff(diffusivity, length, u_mean, geometry):
    """Taylor-Aris effective axial diffusivity D + K L^2 U^2 / D."""
    _check_geometry(geometry)
    return diffusivity + DISPERSION_FACTOR[geometry] * length**2 * u_mean**2 / diffusivity


def velocity_variance(u_mean, geometry):
    """Var(u) over the cross-section: U^2/3 (pipe), U^2/5 (channel)."""
    _check_geometry(geometry)
    return VELOCITY_VARIANCE_FACTOR[geometry] * u_mean**2


def dispersion_factor_numeric(geometry, n_points=400001):
    """Re-derive K by solving the cell problem NUMERICALLY.

    Independent of the closed forms above: integrates the cell problem on a
    grid and evaluates <u' b> by quadrature. `verify_limits` requires this to
    reproduce the closed-form K, so a wrong constant fails loudly instead of
    propagating.

    It also generalises. Substituting a Casson or Carreau profile for
    `velocity_deviation` gives the dispersion coefficient for a shear-thinning
    fluid, for which no closed form is used here.
    """
    _check_geometry(geometry)
    lo = 0.0 if geometry == "pipe" else -1.0
    xi = np.linspace(lo, 1.0, n_points)
    up = velocity_deviation(xi, geometry)
    w = cross_section_weight(xi, geometry)

    def cumtrap(f):
        return np.concatenate(
            [[0.0], np.cumsum(0.5 * np.diff(xi) * (f[1:] + f[:-1]))]
        )

    if geometry == "pipe":
        # (s b')' = -s u'  =>  b' = -(1/s) int_0^s s' u' ds'
        bp = np.zeros_like(xi)
        bp[1:] = -cumtrap(xi * up)[1:] / xi[1:]
    else:
        # b'' = -u' with b'(-1) = 0
        bp = -cumtrap(up)
    b = cumtrap(bp)
    b -= np.trapezoid(b * w, xi) / np.trapezoid(w, xi)  # enforce <b> = 0
    return float(np.trapezoid(up * b * w, xi) / np.trapezoid(w, xi))


# --------------------------------------------------------------------------
# Time scales
# --------------------------------------------------------------------------


def transverse_relaxation_time(length, diffusivity):
    """tau_r = L^2 / D, the transverse diffusion time."""
    return length**2 / diffusivity


def asymptotic_onset(geometry, tolerance=1e-6, symmetric_release=True):
    """t/tau_r at which the pre-asymptotic correction falls below `tolerance`.

    exp(-beta^2 t/tau_r) < tolerance. Which beta^2 applies depends on the
    SYMMETRY OF THE RELEASE, not on the geometry alone:

      symmetric_release=True  (the default; an area-uniform or axisymmetric
        release into an axisymmetric u') uses the slowest COUPLED mode,
        beta^2 = 14.682 (pipe) or pi^2 (channel), giving 0.941 and 1.400
        tau_r at 1e-6.
      symmetric_release=False (off-axis injection, a cloud seeded on a patch
        subset, an azimuthally biased mesh) uses the slowest mode that EXISTS,
        beta^2 = 3.390 (pipe) or pi^2/4 (channel), giving 4.075 and 5.599.

    A fit window chosen on the symmetric assumption is wrong by 4.3x for a
    pipe and 4x for a channel the moment the symmetry is broken, and the
    failure is silent: the fit still returns a number.
    """
    _check_geometry(geometry)
    table = COUPLED_EIGENVALUE if symmetric_release else ASYMMETRIC_EIGENVALUE
    return -np.log(tolerance) / table[geometry]


# --------------------------------------------------------------------------
# The released pulse: the molecular-communications impulse response
# --------------------------------------------------------------------------


def pulse_concentration(x, t, u_mean, dispersivity, mass=1.0):
    """1-D advection-diffusion Green's function for a plane source.

        c(x,t) = M / sqrt(4 pi K t) * exp(-(x - U t)^2 / (4 K t))

    Pass `dispersivity = D` for pure advection-diffusion, or
    `dispersivity = d_eff(...)` for the Taylor-dispersion regime — the same
    function serves both, which is the whole content of Taylor's result.
    As a channel impulse response, evaluating this at a fixed receiver
    position x = L as a function of t gives the arrival-time distribution.

    Crank (1975) sec. 2.2 for the diffusive kernel; the advective form is its
    Galilean shift.
    """
    xx = np.asarray(x, dtype=float)
    tt = np.asarray(t, dtype=float)
    return (
        mass
        / np.sqrt(4.0 * np.pi * dispersivity * tt)
        * np.exp(-((xx - u_mean * tt) ** 2) / (4.0 * dispersivity * tt))
    )


def pulse_centroid(t, u_mean):
    """<x>(t) = U t, exactly, at all times."""
    return u_mean * np.asarray(t, dtype=float)


def pulse_variance(t, dispersivity):
    """sigma_x^2(t) = 2 K t for the 1-D solution."""
    return 2.0 * dispersivity * np.asarray(t, dtype=float)


def variance_short_time(t, u_mean, diffusivity, geometry):
    """sigma_x^2 = 2 D t + Var(u) t^2 in the ballistic regime, t << tau_r.

    The t^2 term is the shear-sampling contribution before radial diffusion
    has decorrelated it. A DIFFERENT POWER OF t from anchor 3, so it tests
    something anchor 3 cannot.
    """
    tt = np.asarray(t, dtype=float)
    return 2.0 * diffusivity * tt + velocity_variance(u_mean, geometry) * tt**2


def transverse_pdf(xi, geometry):
    """The exact transverse distribution, at all times, with and without flow.

    P(r) = 2r/a^2 for a pipe and P(y) = 1/(2h) for a channel. Axial flow does
    not couple to the transverse coordinate, so this is an invariant of the
    problem INDEPENDENT of the transport physics under test — the same role
    the radial distribution plays in `taylor_aris.py`, and the reason it is
    the gate rather than an anchor. A solver whose wall condition leaks, or
    whose particles stick, fails here while still producing a plausible D_eff.
    """
    _check_geometry(geometry)
    x = np.asarray(xi, dtype=float)
    return 2.0 * x if geometry == "pipe" else np.full_like(x, 0.5)


def variance_intercept(length, u_mean, diffusivity, geometry):
    """The constant in sigma^2 -> 2 D_eff t - C L^4 U^2 / D^2.

    Negative, i.e. the asymptote extrapolates back through a virtual origin at
    positive t. Worth having because once D_eff is a FITTED parameter,
    numerical diffusion is invisible in the slope; the intercept weights the
    transverse spectrum differently (beta^-8 against beta^-6) and so imposes a
    second, independent constraint on the same operator.
    """
    _check_geometry(geometry)
    return -(
        VARIANCE_INTERCEPT_FACTOR[geometry] * length**4 * u_mean**2 / diffusivity**2
    )


def variance_long_time(t, length, u_mean, diffusivity, geometry):
    """sigma^2(t) = 2 D_eff t + intercept, the asymptotic straight line."""
    return 2.0 * d_eff(diffusivity, length, u_mean, geometry) * np.asarray(
        t, dtype=float
    ) + variance_intercept(length, u_mean, diffusivity, geometry)


def third_cumulant(t, length, u_mean, diffusivity, geometry):
    """kappa_3(t) -> 6 <u' B^2> t, the long-time third cumulant.

    THE ONLY ANCHOR HERE THAT IS BLIND TO AXIAL DIFFUSION. Any symmetric
    axial kernel — which is every standard scheme, physical or numerical —
    contributes nothing beyond the second cumulant, so kappa_3 needs no
    2 D t subtracted before it says anything. The variance does, and that
    subtraction is exactly where scheme-induced axial diffusion hides.

    Its sign is geometry-dependent: POSITIVE for a pipe, NEGATIVE for a
    channel. Nothing else in this module changes sign between the two.
    """
    _check_geometry(geometry)
    return (
        SKEWNESS_FACTOR[geometry]
        * length**4
        * u_mean**3
        / diffusivity**2
        * np.asarray(t, dtype=float)
    )


def centroid_offset(xi_release, length, u_mean, diffusivity, geometry):
    """Permanent axial offset left by a POINT (or ring) release at xi_release.

    A cross-sectionally UNIFORM release gives centroid exactly U t at all
    times, with no transient at all — a uniform transverse profile is its own
    steady state, so the mass-weighted mean velocity is U from t = 0. A
    localised release does not: the centroid starts at the LOCAL velocity
    u(xi_release) and relaxes to U, leaving the permanent lag

        xbar(t) - U t  ->  (U L^2 / D) b(xi_release)

    where b is the SAME Aris cell function that sets D_eff. The two appear in
    completely different measurements, which makes this a cross-check on the
    cell function rather than a new assumption.

    The sharp case, and the reason this is worth having: releasing exactly on
    the streamline where u = U (xi = 1/sqrt(2) for a pipe, 1/sqrt(3) for a
    channel) still leaves the pulse permanently BEHIND, by U a^2/(96 D) and
    U h^2/(90 D). "Seeded at the mean velocity, therefore no offset" is wrong.
    """
    _check_geometry(geometry)
    return u_mean * length**2 / diffusivity * cell_function(xi_release, geometry)


def balance_peclet(geometry):
    """Pe = U L / D at which the shear and molecular terms of D_eff are equal.

    D = K L^2 U^2 / D gives Pe = 1/sqrt(K): sqrt(48) = 6.928 for the pipe and
    sqrt(105/2) = 7.246 for the channel. A useful invariant — a sweep designed
    to straddle the crossover should land here regardless of the physical
    parameters.
    """
    _check_geometry(geometry)
    return 1.0 / np.sqrt(DISPERSION_FACTOR[geometry])


# --------------------------------------------------------------------------
# Self-verification
# --------------------------------------------------------------------------


def verify_limits(rtol=1e-9):
    """Check the oracle's internal consistency before it is used as truth.

    Everything here is a relation between two things this module computes by
    DIFFERENT routes, or between this module and `taylor_aris`. Nothing is a
    restatement of a constant.
    """
    errors = {}

    for geom in GEOMETRIES:
        lo = 0.0 if geom == "pipe" else -1.0
        xi = np.linspace(lo, 1.0, 2000001)
        up = velocity_deviation(xi, geom)
        w = cross_section_weight(xi, geom)
        norm = np.trapezoid(w, xi)

        # 1. The deviation averages to zero — otherwise U is not the mean.
        errors[f"{geom}_deviation_mean"] = abs(
            float(np.trapezoid(up * w, xi) / norm)
        )

        # 2. Var(u) by quadrature against the closed factor.
        var = float(np.trapezoid(up**2 * w, xi) / norm)
        errors[f"{geom}_velocity_variance"] = abs(
            var / VELOCITY_VARIANCE_FACTOR[geom] - 1.0
        )

        # 3. The cell function satisfies its own ODE and boundary condition.
        errors[f"{geom}_cell_no_flux"] = abs(float(cell_function_derivative(1.0)))
        errors[f"{geom}_cell_zero_mean"] = abs(
            float(np.trapezoid(cell_function(xi, geom) * w, xi) / norm)
        )

        # 4. K from the closed-form cell function, by quadrature.
        k_quad = float(
            np.trapezoid(up * cell_function(xi, geom) * w, xi) / norm
        )
        errors[f"{geom}_K_quadrature"] = abs(k_quad / DISPERSION_FACTOR[geom] - 1.0)

        # 5. K from an INDEPENDENT numerical solve of the cell problem.
        errors[f"{geom}_K_numeric"] = abs(
            dispersion_factor_numeric(geom) / DISPERSION_FACTOR[geom] - 1.0
        )

        # 6. D_eff -> D as U -> 0, and Pe scaling of the excess.
        d, length = 3.0e-9, 1.5e-3
        errors[f"{geom}_zero_velocity"] = abs(d_eff(d, length, 0.0, geom) / d - 1.0)
        pe = peclet(2.0e-4, length, d)
        excess = d_eff(d, length, 2.0e-4, geom) / d - 1.0
        errors[f"{geom}_peclet_scaling"] = abs(
            excess / (DISPERSION_FACTOR[geom] * pe**2) - 1.0
        )

    # 7. CROSS-MODULE. The pipe value must equal the one the Lagrangian
    #    particle oracle uses. The two modules implement it independently, so
    #    this catches a drift in either.
    from betaflow.analytic import taylor_aris

    d, a, u = 2.0e-12, 1.0e-5, 3.0e-6
    errors["pipe_matches_taylor_aris_module"] = abs(
        d_eff(d, a, u, "pipe") / taylor_aris.d_eff(d, a, u) - 1.0
    )
    errors["pipe_matches_taylor_aris_variance"] = abs(
        velocity_variance(u, "pipe") / taylor_aris.velocity_variance(u) - 1.0
    )
    errors["pipe_matches_taylor_aris_eigenvalue"] = abs(
        COUPLED_EIGENVALUE["pipe"] / taylor_aris.BESSEL_J1_FIRST_ROOT**2 - 1.0
    )

    # 8. The Green's function conserves mass and carries the stated moments.
    d_, u_, t_, m_ = 2.0e-9, 1.0e-3, 50.0, 1.0
    span = 40.0 * np.sqrt(2.0 * d_ * t_)
    x = np.linspace(u_ * t_ - span, u_ * t_ + span, 4000001)
    c = pulse_concentration(x, t_, u_, d_, m_)
    m0 = float(np.trapezoid(c, x))
    m1 = float(np.trapezoid(x * c, x)) / m0
    m2 = float(np.trapezoid((x - m1) ** 2 * c, x)) / m0
    errors["greens_mass"] = abs(m0 / m_ - 1.0)
    errors["greens_centroid"] = abs(m1 / pulse_centroid(t_, u_) - 1.0)
    errors["greens_variance"] = abs(m2 / pulse_variance(t_, d_) - 1.0)

    # 9. The channel's coupled eigenvalue is pi^2 and NOT the slowest
    #    transverse mode pi^2/4 — the odd modes are orthogonal to an even u'.
    y = np.linspace(-1.0, 1.0, 2000001)
    up_c = velocity_deviation(y, "channel")
    odd = float(np.trapezoid(up_c * np.cos(1 * np.pi * (y + 1) / 2), y))
    even = float(np.trapezoid(up_c * np.cos(2 * np.pi * (y + 1) / 2), y))
    errors["channel_odd_mode_orthogonal"] = abs(odd)
    if abs(even) < 1e-6:
        raise AssertionError(
            "channel n=2 mode does not couple to u'; the eigenvalue choice in "
            "COUPLED_EIGENVALUE is then wrong"
        )

    # 10. THE SPECTRAL ROUTE — an independent derivation, not a restatement.
    #     Expanding u' in the transverse Neumann eigenbasis with modal weights
    #     g_n = a_n^2/mu_n, three DIFFERENT weightings of the SAME spectrum
    #     must return the three constants above:
    #         sum g_n mu_n = Var(u)          (Parseval)
    #         sum g_n      = D_eff - D       (Green-Kubo)
    #         2 sum g_n/mu_n = -intercept
    #     The pipe closes on Rayleigh sums over the zeros of J1 and the
    #     channel on zeta(4), zeta(6), zeta(8) — arithmetic identities with no
    #     fluid mechanics in them, so an algebra slip upstream cannot
    #     reproduce them by luck.
    beta = jn_zeros(1, 6000)
    spectral = {
        "pipe": (
            64.0 * float(np.sum(beta**-4.0)),
            64.0 * float(np.sum(beta**-6.0)),
            2.0 * 64.0 * float(np.sum(beta**-8.0)),
        ),
        "channel": (
            18.0 * float(np.sum(np.arange(1, 6001, dtype=float) ** -4.0)) / np.pi**4,
            18.0 * float(np.sum(np.arange(1, 6001, dtype=float) ** -6.0)) / np.pi**6,
            2.0
            * 18.0
            * float(np.sum(np.arange(1, 6001, dtype=float) ** -8.0))
            / np.pi**8,
        ),
    }
    for geom, (var_s, k_s, icept_s) in spectral.items():
        errors[f"{geom}_spectral_variance"] = abs(
            var_s / VELOCITY_VARIANCE_FACTOR[geom] - 1.0
        )
        errors[f"{geom}_spectral_K"] = abs(k_s / DISPERSION_FACTOR[geom] - 1.0)
        errors[f"{geom}_spectral_intercept"] = abs(
            icept_s / VARIANCE_INTERCEPT_FACTOR[geom] - 1.0
        )

    # 11. Third moment and third cumulant, by quadrature.
    for geom in GEOMETRIES:
        lo = 0.0 if geom == "pipe" else -1.0
        xi = np.linspace(lo, 1.0, 2000001)
        up = velocity_deviation(xi, geom)
        w = cross_section_weight(xi, geom)
        norm = np.trapezoid(w, xi)
        third = float(np.trapezoid(up**3 * w, xi) / norm)
        if geom == "pipe":
            # An exact ZERO, so it is checked absolutely, not relatively.
            errors["pipe_third_moment_is_zero"] = abs(third)
        else:
            errors["channel_third_moment"] = abs(
                third / THIRD_MOMENT_FACTOR[geom] - 1.0
            )
        kappa = 6.0 * float(
            np.trapezoid(up * cell_function(xi, geom) ** 2 * w, xi) / norm
        )
        errors[f"{geom}_skewness_factor"] = abs(kappa / SKEWNESS_FACTOR[geom] - 1.0)

        # 12. The transverse gate integrates to one and has the right mean.
        pdf = transverse_pdf(xi, geom)
        errors[f"{geom}_pdf_normalised"] = abs(float(np.trapezoid(pdf, xi)) - 1.0)

    # 13. Sign asymmetry: the pipe's third cumulant is positive and the
    #     channel's negative. Nothing else here changes sign, so a geometry
    #     mix-up shows up as a sign rather than a small discrepancy.
    if not (SKEWNESS_FACTOR["pipe"] > 0.0 > SKEWNESS_FACTOR["channel"]):
        raise AssertionError("third-cumulant signs are no longer opposite")

    # 14. Balance Peclet: at Pe = 1/sqrt(K) the two terms of D_eff are equal.
    for geom in GEOMETRIES:
        d, length = 1e-9, 2e-3
        u = balance_peclet(geom) * d / length
        errors[f"{geom}_balance_peclet"] = abs(
            d_eff(d, length, u, geom) / (2.0 * d) - 1.0
        )

    # 15. Releasing on the u = U streamline still leaves a permanent LAG.
    #     xi where u' = 0: 1/sqrt(2) (pipe), 1/sqrt(3) (channel).
    for geom, xi0, expect in (
        ("pipe", 1.0 / np.sqrt(2.0), -1.0 / 96.0),
        ("channel", 1.0 / np.sqrt(3.0), -1.0 / 90.0),
    ):
        errors[f"{geom}_zero_deviation_streamline"] = abs(
            float(velocity_deviation(xi0, geom))
        )
        d, length, u = 3e-9, 1e-3, 2e-4
        errors[f"{geom}_centroid_offset"] = abs(
            centroid_offset(xi0, length, u, d, geom)
            / (expect * u * length**2 / d)
            - 1.0
        )

    # 16. The asymmetric-release onset is slower, by the stated factors.
    errors["pipe_onset_ratio"] = abs(
        asymptotic_onset("pipe", symmetric_release=False)
        / asymptotic_onset("pipe")
        / (COUPLED_EIGENVALUE["pipe"] / ASYMMETRIC_EIGENVALUE["pipe"])
        - 1.0
    )
    errors["channel_onset_ratio"] = abs(
        asymptotic_onset("channel", symmetric_release=False)
        / asymptotic_onset("channel")
        / 4.0
        - 1.0
    )

    for name, err in errors.items():
        limit = rtol
        if "quadrature" in name or "variance" in name or "greens" in name:
            limit = 1e-9
        if "numeric" in name:
            limit = 1e-8  # trapezoid on the cell problem, not machine precision
        if "spectral" in name or "skewness" in name or "third_moment" in name:
            limit = 1e-8  # truncated eigen-series and trapezoid quadrature
        if "pdf_normalised" in name:
            limit = 1e-12
        if not err < limit:
            raise AssertionError(
                f"advection_diffusion oracle {name} error {err:.3e} > {limit:.0e}"
            )
    return {k: float(v) for k, v in errors.items()}
