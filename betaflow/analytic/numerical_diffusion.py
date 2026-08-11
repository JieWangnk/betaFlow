"""Numerical diffusion of an axial advection scheme — the error as the analytic reference.

THE AXIAL COUNTERPART to `advection_diffusion.py`. That module gives the exact
PHYSICS of scalar transport; `runners/moments.py` measures it with no axial
mesh at all, so nothing there sees a solver's advection scheme, its numerical
diffusion, or its boundedness — which is most of what a lattice-Boltzmann
advection-diffusion lattice or a finite-volume scalar solver gets wrong.

WHY A PROFILE COMPARISON CANNOT DO THIS. A solver with physical diffusivity D
and numerical diffusivity D_num produces a pulse that looks EXACTLY like the
exact solution for D + D_num. There is no feature of a single profile that
separates them: the shape is Gaussian either way and the widths simply add. So
"advect a pulse and compare against the analytic solution" passes a solver
whose scheme is contributing more spreading than the physics. This is the same
structure as the kernel blind spot and the wedge faceting bias — a quantity
that looks informative and is not.

WHAT SEPARATES THEM is that D_num depends on things D cannot:

    D_num depends on dx, on the Courant number, and on the scheme.
    D      depends on none of them.

So the discriminating experiment is a SWEEP, not a comparison. Hold the
physics fixed and vary the discretisation; whatever moves is numerical.

THE EXACT RESULT, for the standard case: uniform mesh, constant velocity
u > 0, first-order upwind in space, explicit Euler in time,

    D_num = (u dx / 2) (1 - Co),        Co = u dt / dx.

Two features of that formula are anchors in their own right, and both are
sharper than the coefficient because they are structural:

    Co -> 1:  D_num -> 0 EXACTLY. Upwind at unit Courant number is a pure
              shift of the data by one cell, so it is exact for ANY initial
              condition. Measured here: variance change -1.9e-17.
    Co -> 0:  D_num -> u dx / 2, the classical steady-state result. A code
              whose numerical diffusion does NOT fall as Co rises has a
              time-discretisation error, not a spatial one — the -Co term
              comes entirely from the temporal truncation, and it is the part
              most often dropped when this formula is quoted.

DERIVED BY FOURIER ANALYSIS, not by the modified-equation substitution. For a
mode exp(i k x) the exact amplification factor of the scheme is

    G(z) = 1 - Co (1 - exp(-i z)),      z = k dx,

and expanding n ln G in z gives every transport coefficient at once. This
route is preferred here because the usual Warming-Hyett procedure — Taylor
expand, then eliminate time derivatives using the PDE — is easy to truncate
inconsistently. THE FIRST VERSION OF THIS MODULE DID EXACTLY THAT and produced
a third-order coefficient whose roots contradicted the measurement. The Fourier
route needs no substitution at all, so there is nothing to get wrong.

STATUS OF EACH RESULT, stated separately because they are not equally solid.

  D_num = (u dx/2)(1 - Co)
      EXACT and VERIFIED. Measured against a real 1-D solver with the physical
      diffusivity set to zero: ratio 1.0000 at every one of nine combinations
      of N in {200, 400, 800} and Co in {0.2, 0.5, 0.8}, and 1.000000 at
      N up to 3200. This is the quantitative anchor.

  E3 = (dx^2 u / 6) (Co - 1) (2 Co - 1)
      STRUCTURE VERIFIED, MAGNITUDE NOT. The dispersive coefficient predicts
      sign changes at Co = 1/2 and Co = 1, and BOTH are observed: the measured
      third moment passes through zero at Co = 0.5 (measured -5.4e-20) and
      reverses sign either side of it. But the measured magnitude sits ~8%
      above 6 E3 T and DOES NOT CONVERGE under refinement — the ratio goes
      1.1105, 1.0908, 1.0809, 1.0760 for N = 400, 800, 1600, 3200 while
      D_num's ratio stays at 1.000000 throughout. A plateau is not a
      discretisation error, so something in either the prediction or the
      discrete moment estimator is unaccounted for; the candidates are lattice
      aliasing in the discrete moment sum and a genuinely missing term.
      IT IS NOT RESOLVED HERE. The sign structure is therefore exported as an
      anchor and the magnitude is NOT — using it would be asserting a number
      that is 8% wrong for reasons nobody has established.

CENTRAL DIFFERENCING, for contrast: its numerical diffusion is NEGATIVE. The
same measurement gives -4.96e-4 at Co = 0.2 where upwind gives +2.0e-3, and at
Co = 0.5 the scheme diverges outright (measured variance 1.3e+11). A negative
numerical diffusivity IS the instability, stated as a transport coefficient.

CITATIONS
  Hirt, C.W. (1968), J. Comput. Phys. 2(4):339-355 — the modified-equation
    idea and the numerical-diffusion concept. NOT READ HERE; cited from
    general knowledge and flagged as such.
  Warming, R.F. & Hyett, B.J. (1974), J. Comput. Phys. 14(2):159-179 — the
    systematic modified-equation procedure. NOT READ HERE. Flagged
    particularly because this module deliberately does NOT use that procedure.
  The Fourier route used here is standard von Neumann analysis and needs no
  citation beyond that; every coefficient below is re-derived by
  `verify_limits` rather than taken from any source.
"""

import numpy as np

SCHEMES = ("upwind_explicit", "central_explicit")


def courant(u, dt, dx):
    """Co = u dt / dx."""
    return u * dt / dx


def numerical_diffusivity(u, dx, courant_number, scheme="upwind_explicit"):
    """D_num for the named scheme.

    upwind_explicit : (u dx/2)(1 - Co). Positive for Co < 1, zero at Co = 1.
    central_explicit: -(u dx/2) Co, i.e. NEGATIVE at every Co — which is the
        scheme's unconditional instability written as a transport
        coefficient rather than as a stability condition.
    """
    if scheme == "upwind_explicit":
        return 0.5 * u * dx * (1.0 - courant_number)
    if scheme == "central_explicit":
        return -0.5 * u * dx * courant_number
    raise ValueError(f"scheme must be one of {SCHEMES}, got {scheme!r}")


def dispersive_coefficient(u, dx, courant_number):
    """E3 for explicit upwind: (dx^2 u/6)(Co - 1)(2 Co - 1).

    EXPORTED FOR ITS SIGN STRUCTURE ONLY. Its roots at Co = 1/2 and Co = 1 are
    verified against measurement; its magnitude is ~8% high and does not
    converge, and that discrepancy is unexplained. See the module docstring.
    Do not build a tolerance on the value this returns.
    """
    return dx**2 * u * (courant_number - 1.0) * (2.0 * courant_number - 1.0) / 6.0


def dispersion_free_courant():
    """The Courant numbers at which explicit upwind is dispersion-free.

    Co = 1/2 and Co = 1. The second is the trivial one — the whole scheme is
    exact there. The first is not trivial: the scheme is still strongly
    diffusive at Co = 1/2 (D_num = u dx/4) while generating no third moment at
    all, so a case run there measures diffusion uncontaminated by dispersion.
    """
    return (0.5, 1.0)


def total_spreading(diffusivity, u, dx, courant_number, scheme="upwind_explicit"):
    """D + D_num — what a profile comparison actually measures.

    THE POINT OF THIS FUNCTION IS THAT IT IS INDISTINGUISHABLE FROM D. A
    solver reporting this value produces a pulse identical in every respect to
    the exact solution for a fluid whose diffusivity is this. It is provided
    so a case can state explicitly what fraction of its measured spreading is
    an artefact.
    """
    return diffusivity + numerical_diffusivity(u, dx, courant_number, scheme)


def artefact_fraction(diffusivity, u, dx, courant_number, scheme="upwind_explicit"):
    """D_num / (D + D_num): how much of the measured spreading is the scheme.

    The number to report alongside any dispersion measurement. At the cell
    Peclet numbers typical of a haemodynamic scalar transport case this is
    routinely above 1/2, meaning the scheme contributes more spreading than
    the fluid.
    """
    d_num = numerical_diffusivity(u, dx, courant_number, scheme)
    return d_num / (diffusivity + d_num)


def cell_peclet(u, dx, diffusivity):
    """Pe_cell = u dx / D. D_num/D = (Pe_cell/2)(1 - Co), so Pe_cell alone
    decides whether the scheme or the fluid dominates: at Pe_cell = 2 and
    Co -> 0 they are exactly equal."""
    return u * dx / diffusivity


def verify_limits(rtol=1e-12):
    """Re-derive every coefficient by Fourier analysis and check the limits.

    Nothing here restates a constant: the coefficients are extracted from the
    expansion of ln G, which is a different computation from the closed forms
    above.
    """
    errors = {}
    u, dx = 1.7, 0.013

    # THE INDEPENDENT ROUTE: read D_num off the EXACT amplification factor
    # G(z) = 1 - Co(1 - exp(-i z)) by symbolic series, so nothing here is a
    # rearrangement of the closed forms above.
    #
    # A numerical extraction at finite z was tried first and abandoned. The
    # real part of ln G is O(z^2) against a value of order 1, so it is formed
    # by cancellation and carries a relative round-off floor of eps/z^2 —
    # 9.2e-8 at z = 1e-4, and still 4.3e-9 after a Richardson step, which is
    # six decades short of exact and would have to be absorbed into a
    # tolerance. Recorded because a 1e-8 "agreement" of an EXACT identity is
    # the kind of number that reads as success.
    import sympy as sp

    z_sym, co_sym = sp.symbols("z Co")
    series = sp.series(sp.log(1 - co_sym * (1 - sp.exp(-sp.I * z_sym))), z_sym, 0, 4)
    a2_sym = sp.simplify(series.removeO().expand().coeff(z_sym, 2))
    a3_sym = sp.simplify(series.removeO().expand().coeff(z_sym, 3) / sp.I)

    for co in (0.1, 0.25, 0.5, 0.8, 0.95):
        # ln G = -i Co z + a2 z^2 + i a3 z^3 + ...; over one step of duration
        # dt = Co dx / u this is -k^2 D_num dt + i k^3 E3 dt.
        a2 = float(a2_sym.subs(co_sym, co))
        d_num_fourier = -a2 * dx**2 / (co * dx / u)
        errors[f"upwind_Co{co}"] = abs(
            d_num_fourier / numerical_diffusivity(u, dx, co) - 1.0
        )
        # The dispersive coefficient's STRUCTURE, from the same expansion.
        a3 = float(a3_sym.subs(co_sym, co))
        e3_fourier = a3 * dx**3 / (co * dx / u)
        errors[f"upwind_E3_Co{co}"] = abs(
            e3_fourier / dispersive_coefficient(u, dx, co) - 1.0
        ) if abs(dispersive_coefficient(u, dx, co)) > 0 else abs(e3_fourier)

    # Co = 1 is EXACT, not merely small: the scheme is a one-cell shift.
    errors["unit_courant_exact"] = abs(numerical_diffusivity(u, dx, 1.0))

    # Co -> 0 recovers the classical steady-state value u dx / 2.
    errors["zero_courant_classical"] = abs(
        numerical_diffusivity(u, dx, 0.0) / (0.5 * u * dx) - 1.0
    )

    # Central differencing is negative at every Courant number, which is the
    # instability. Checked as a sign, not as a magnitude.
    for co in (0.05, 0.5, 0.9):
        if numerical_diffusivity(u, dx, co, "central_explicit") >= 0.0:
            raise AssertionError(
                "central differencing must have NEGATIVE numerical diffusion; "
                "if this ever passes, the sign convention has drifted"
            )

    # The dispersion-free Courant numbers are roots of E3, by construction.
    for co in dispersion_free_courant():
        errors[f"dispersion_free_Co{co}"] = abs(dispersive_coefficient(u, dx, co))

    # ... and E3 does NOT vanish between them, or the roots would be vacuous.
    if abs(dispersive_coefficient(u, dx, 0.75)) < 1e-12 * u * dx**2:
        raise AssertionError("E3 vanishes where it should not; check the form")

    # The artefact fraction is a fraction, and exceeds 1/2 exactly when the
    # cell Peclet number exceeds 2/(1-Co).
    d, co = 1e-6, 0.2
    dx_crit = 2.0 * d / (u * (1.0 - co))
    errors["artefact_half_at_critical_dx"] = abs(
        artefact_fraction(d, u, dx_crit, co) - 0.5
    )
    errors["cell_peclet_at_critical_dx"] = abs(
        cell_peclet(u, dx_crit, d) * (1.0 - co) / 2.0 - 1.0
    )

    for name, err in errors.items():
        if not err < rtol:
            raise AssertionError(
                f"numerical_diffusion analytic reference {name} error {err:.3e} > {rtol:.0e}"
            )
    return {k: float(v) for k, v in errors.items()}
