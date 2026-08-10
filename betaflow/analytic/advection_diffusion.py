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

WHICH TRANSVERSE MODE ACTUALLY MATTERS. For no-flux walls the channel's
transverse eigenfunctions are cos(n pi (y+h) / 2h) with eigenvalues
(n pi / 2h)^2, so the slowest nonzero mode is pi^2/(4 h^2). That is NOT the
rate that governs the approach to Taylor dispersion. u'(y) is EVEN in y, so it
is orthogonal to every odd mode: the overlap <u', phi_n> vanishes identically
for n = 1, 3, 5, ... and the slowest COUPLED mode is n = 2, giving pi^2/h^2.
The pipe has no such cancellation because its J0 modes are all even, so its
rate is the first J1 root squared, 14.682. Quoting the slowest EXISTING mode
would make the channel look twice as slow to reach the asymptote as it is.

CITATIONS
  Taylor, G.I. (1953), Proc. R. Soc. A 219:186-203 — dispersion in a pipe.
  Aris, R. (1956), Proc. R. Soc. A 235:67-77 — method of moments, general
    cross-section, and the D + a^2 U^2/(48 D) form used here.
  Crank, J. (1975), "The Mathematics of Diffusion", 2nd ed., OUP, sec. 2.2 —
    the free-space Green's function of anchor 2.
  Carslaw, H.S. & Jaeger, J.C. (1959), "Conduction of Heat in Solids", 2nd
    ed., OUP — the same solution and the Neumann eigenfunction expansion.

  UNVERIFIED, and flagged rather than dressed up: the plane-channel constant
  2/105 is standardly attributed to Wooding (1960), J. Fluid Mech. 7:501-515,
  but that paper has NOT been read here. The constant is therefore used on the
  strength of the derivation above and of `verify_limits`, which re-derives it
  numerically and separately confirms that the identical route returns Aris's
  published 1/48 for the pipe. Treat the attribution as a lead to check, not
  as a source. The pipe constant needs no such caveat: it is Aris's own.
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


def asymptotic_onset(geometry, tolerance=1e-6):
    """t/tau_r at which the pre-asymptotic correction falls below `tolerance`.

    exp(-beta^2 t/tau_r) < tolerance. Returns 0.941 for the pipe and 1.400 for
    the channel at 1e-6. Both are O(1), so fitting a dispersion coefficient
    from 2 tau_r onward is safe in either geometry — but the channel's margin
    is 1.4x rather than the pipe's 2.1x, because its coupled mode decays more
    slowly (pi^2 against 14.68). A fit window copied from the pipe case
    without re-checking would be the obvious way to get this wrong.
    """
    _check_geometry(geometry)
    return -np.log(tolerance) / COUPLED_EIGENVALUE[geometry]


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

    for name, err in errors.items():
        limit = rtol
        if "quadrature" in name or "variance" in name or "greens" in name:
            limit = 1e-9
        if "numeric" in name:
            limit = 1e-8  # trapezoid on the cell problem, not machine precision
        if not err < limit:
            raise AssertionError(
                f"advection_diffusion oracle {name} error {err:.3e} > {limit:.0e}"
            )
    return {k: float(v) for k, v in errors.items()}
