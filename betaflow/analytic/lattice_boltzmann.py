"""Lattice-Boltzmann transport coefficients and lattice constants.

THE THIRD LEG of the Eulerian scalar ladder. `advection_diffusion.py` gives the
exact PHYSICS, `numerical_diffusion.py` gives a finite-volume scheme's own
error, and this module gives the relations a lattice-Boltzmann code must
satisfy by construction — the ones that are exact statements about the method
rather than about the flow.

WHAT IS DERIVED HERE VERSUS WHAT IS CITED. Everything in this module that is
pure algebra is COMPUTED from the velocity sets, not remembered: the sound
speeds, the weights' normalisation, and the moment-isotropy conditions are all
re-derived in `verify_limits` from the discrete velocities themselves. The
Chapman-Enskog result D = c_s^2 (tau - 1/2) dt is NOT derivable from the
velocity set alone and is cited; its CONSEQUENCES are then checked here.

  1. THE RELAXATION-DIFFUSIVITY RELATION.

         D = c_s^2 (tau - 1/2) dt

     The -1/2 is the whole reason the method is second-order rather than
     first: it is the discrete-time correction that appears when the
     collision operator is applied over a finite step rather than
     continuously. A code that drops it and writes D = c_s^2 tau dt is not
     slightly wrong — it is wrong by a factor that DEPENDS ON TAU:

         tau = 2.00  ->  1.33x too large
         tau = 1.00  ->  2.00x
         tau = 0.60  ->  6.00x
         tau = 0.51  ->  51.0x

     That structure is what makes it dangerous. A code calibrated at large tau
     looks acceptable and then fails by orders of magnitude in exactly the
     regime a low-diffusivity solute demands, and the failure is silent
     because the concentration field still looks like a diffusing pulse. This
     is the same shape as the numerical-diffusion blind spot: the wrong answer
     is a plausible member of the same family of solutions.

     Two limits follow immediately and are worth asserting on their own:
     tau -> 1/2 gives D -> 0, and tau < 1/2 gives a NEGATIVE diffusivity — the
     lattice statement of the same instability that central differencing
     expresses as a negative numerical diffusivity in `numerical_diffusion`.

  2. THE VELOCITY SETS, AND THE REAL SHAPE OF THE D3Q7 TRAP.

     The first version of this module stated "c_s^2 = 1/4 for D3Q7" as a
     lattice constant. That is true for the weights tabulated below and WRONG
     as a statement about the name: for the reduced scalar lattices c_s^2 is
     a WEIGHT FAMILY, not a constant. With rest weight w_0 = 1 - omega,

         D3Q7:  w_{1..6} = omega/6,  c_s^2 = omega/3,   0 < omega < 1
         D2Q5:  w_{1..4} = omega/4,  c_s^2 = omega/2

     so the common textbook D3Q7 (w_0 = 1/4, omega = 3/4) gives exactly 1/4,
     a rest-free D3Q7 (omega = 1) gives 1/3, and BOTH published values are
     correct for different weights — which is where the literature's 1/4-vs-
     1/3 confusion comes from. The trap is therefore sharper than a wrong
     constant: an analytic reference keyed by LATTICE NAME is wrong by construction.
     OpenLB's own user guide carries the proof: its general D2Q9 descriptor
     has cs2 = 1/3 while its D2Q5 thermal MRT model declares c_s^2 = 0.2
     (w_0 = 3/5, omega = 2/5 -> omega/2 = 1/5). Same name, different constant.
     Read c_s^2 off the weights the solver actually uses — which is what
     `sound_speed_squared_from_weights` does and `verify_limits` checks.

     The full sets D2Q9, D3Q19, D3Q27 (and D1Q3) have no such freedom at
     second order: their tabulated weights give c_s^2 = 1/3.

     For the TARGET CODE: OpenLB's D3Q7 particle-ADE lattice sets
     omega_ADE = [4 D_lattice + 1/2]^{-1} (user guide 1.7r0, Eq. 4.58), i.e.
     tau = 4 D + 1/2, which inverts to D = (1/4)(tau - 1/2) — documentary
     evidence that OpenLB's D3Q7 uses c_s^2 = 1/4. The guide never prints the
     D3Q7 weights, so confirm against src/dynamics/latticeDescriptors.h
     (cs2<3,7>) when a checkout is available.

  3. WHICH SETS CAN CARRY WHICH EQUATION.

     An advection-diffusion lattice needs only the SECOND moment to be
     isotropic. Navier-Stokes additionally needs the fourth,

         sum_i w_i c_ia c_ib c_ig c_id = c_s^4 (d_ab d_gd + d_ag d_bd + d_ad d_bg)

     i.e. M4_xxxx = 3 c_s^4 and M4_xxyy = c_s^4. Computing both from the
     velocity sets gives:

         D2Q9, D3Q19, D3Q27   both conditions hold   -> fluid or scalar
         D2Q5                 M4_xxyy = 0, needs 1/9 -> SCALAR ONLY
         D3Q7                 M4_xxxx = 1/4 needs 3/16,
                              M4_xxyy = 0   needs 1/16 -> SCALAR ONLY

     So the reduced sets are not merely cheaper, they are incapable of
     carrying momentum, and a coupled simulation must use a full set for the
     fluid lattice whatever it uses for the scalar.

  4. THE ADE DIRICHLET WALL SLIP, AND THE MAGIC PARAMETER — RESOLVED.

     The first version of this module listed the scalar-lattice wall
     condition as an open gap. It now has a closed form, from the MRT
     convection-diffusion analysis at arXiv:1603.09577 (read in full by the
     research pipeline; Eqs. 71-73), with every identity re-verified here in
     sympy. For steady 1-D advection-diffusion with anti-bounce-back
     Dirichlet walls, the BGK solution is the exact one plus a UNIFORM
     spurious offset

         phi_s = (dphi / (12 N^2)) * [4 (2/s - 1)^2 - 3],      s = 1/tau,

     and the bracket is IDENTICALLY 16 [ (tau - 1/2)^2 - 3/16 ]. So the slip
     vanishes exactly when Lambda = (tau - 1/2)^2 = 3/16 — Ginzburg's magic
     parameter, emerging from a paper that never names it — at the single
     value tau = 1/2 + sqrt(3)/4 = 0.93301 (s = 4(2 - sqrt(3)) = 1.07180).

     THE NUANCE THAT MATTERS FOR THIS REPO: the slip scales as 1/N^2. It is
     NOT the analogue of the wedge faceting bias — it converges away at
     second order, inflating the error constant rather than destroying the
     order. A refinement study DOES see it; what a refinement study cannot
     see is that a single tuning choice removes it entirely.

     For MRT, zero slip requires s1 = 8(s3 - 2)/(s3 - 8), which satisfies
     (1/s3 - 1/2)(1/s1 - 1/2) = 3/16 identically. THE PUBLISHED EQ. (73) HAS
     A SIGN ERROR — it prints 8(s3 - 2)/(8 - s3), which yields a NEGATIVE
     (unphysical) s1 for every admissible s3 < 8. Verified two ways: the
     corrected form reproduces Lambda = 3/16, and setting s1 = s3 in the
     paper's own Eq. (72b) recovers its BGK formula while the printed (73)
     does not. Recorded alongside Beard (2001) in `advection_diffusion`: a
     published wrong value for exactly the constant under test.

  5. TRT STRUCTURE, from Ginzburg (2012), Commun. Comput. Phys. (abstract
     read; title verified): two DIFFERENT relations between the two
     relaxation rates annihilate the third-order (advection) and fourth-order
     (pure diffusion) truncation errors, for any linear equilibrium and any
     velocity set — so no single magic value kills both, and the leading
     advection-diffusion error CANNOT be removed by relaxation tuning alone;
     the scheme carries an intrinsic fourth-order numerical diffusion. An
     analytic reference assuming "choose Lambda and numerical diffusion vanishes" is
     wrong by construction. The same paper supplies an exact three-time-level
     recurrence form of the TRT update — an independent self-verification
     route requiring no Chapman-Enskog expansion.

  6. THE O(Ma^2) ADVECTION ERROR — RESOLVED BY DERIVATION, for the linear
     (first-order) equilibrium that OpenLB's ADE lattice uses. The BGK ADE
     update with a linear equilibrium is LINEAR in the distributions, so its
     exact amplification matrix G(k) is computable — the same von Neumann
     route that produced the upwind coefficients in `numerical_diffusion`.
     Expanding the conserved eigenvalue's logarithm in k gives, exactly, for
     flow and wavevector along a lattice axis:

         first-order equilibrium:   u_eff = u,
                                    D_eff = (c_s^2 - u^2)(tau - 1/2)
                                          = D (1 - Ma^2)
         second-order equilibrium:  D_eff = c_s^2 (tau - 1/2), the u^2 term
                                    cancelled identically — WHEN c_s^2 takes
                                    its standard value.

     So the Ma^2 coefficient is exactly -1: the scheme's advection DEPLETES
     its own diffusivity by the squared lattice Mach number, and a code
     running a solute at u = 0.3 lattice units under-diffuses by 27%
     (Ma^2 = 3 u^2 on a c_s^2 = 1/3 lattice) with a concentration field that
     looks entirely plausible.

     ON THE REDUCED-WEIGHT FAMILIES THE SECOND-ORDER FIX FAILS. For the
     D2Q5(omega) family the same derivation gives a residual relative
     diffusivity error of -u^2 (3 omega - 2)/omega^2 even WITH the
     second-order equilibrium term: zero only at omega = 2/3 (the
     c_s^2 = 1/3 member), and +5 u^2 — an OVERCORRECTION — at OpenLB's
     thermal weights omega = 2/5. The standard second-order term is built
     for the standard moments, and the reduced weights do not have them.

     The dispersive term closes the circle with the research findings:
     E3 = 2 u^3 (Lambda - 1/12) for the first-order D1Q3 scheme, vanishing
     at Ginzburg's ADVECTION magic value Lambda = 1/12 — a different value
     from the wall's 3/16, confirming by direct derivation the TRT claim
     that no single Lambda kills both errors.

     VERIFIED ON AN ACTUAL LATTICE, ratio 1.000000000 (worst 3.7e-9) across
     tau in {0.6, 1, 2} x u in {0, 0.1, 0.3}, both equilibria, and the
     D2Q5 omega = 2/5 overcorrection — after one instrument failure worth
     recording: the first measurement showed a CONSTANT 0.998667 ratio
     everywhere, which was attributed to finite-pulse-width k^4 truncation,
     a plausible physical mechanism. Doubling the pulse width twice refuted
     the attribution (the deficit did not move), and the true cause was an
     off-by-one — 749 elapsed steps divided by 750. The plausible physics
     story was pasted onto a bookkeeping bug, and only the named-alternative
     test caught it.

WHAT THIS MODULE STILL DOES NOT CLAIM. The Ma^2 result above is exact for
flow along a lattice axis on D1Q3 and the D2Q5(omega) family with BGK; other
sets and diagonal flow directions are conjectured to follow the same law and
are NOT derived here. No published statement of the -1 coefficient was
located even by a 24-source search, so the derivation above is its own
anchor — flagged as this-repo-derived rather than literature-backed. The
tau-dependent wall position of MOMENTUM bounce-back remains open: the one
claim tying it to a readable source failed adversarial verification (0-3),
so this module still cites the anchors as NOT READ and states no
coefficient.

CITATIONS
  arXiv:1603.09577 (MRT for nonlinear convection-diffusion; Eqs. 26, 30b,
    71-73). READ IN FULL by the research pipeline; the slip formula, the
    magic-parameter identity and the Eq. (73) sign error were re-verified
    symbolically in this repo. Also the source for the generalised relation
    kappa = d c_s^2 (1/s - 1/2) dt, whose extra free parameter d (default 1)
    means tau <-> D is not unique unless d = 1 is asserted.
  Liu, Q. & He, Y.-L. (arXiv:1801.00504, review, Table 1). READ by the
    research pipeline, table extracted verbatim and re-verified here: the
    omega-families for D2Q5/D3Q7 and the -1/2 relation for the scalar lattice
    in diffusivity and conductivity forms, surviving into MRT as
    alpha = c_sT^2 (1/sigma - 1/2) dt.
  OpenLB user guide 1.7r0 (openlb.net, June 2024). READ by the research
    pipeline: Eq. 4.58 (omega_ADE = [4D + 1/2]^{-1}, implying c_s^2 = 1/4 for
    its D3Q7 by inversion — an inference, not a printed constant), Eq. 4.21
    (first-order ADE equilibrium), and c_s^2 = 0.2 for its D2Q5 thermal MRT.
  Ginzburg, I. (2012), "Truncation Errors, Exact and Heuristic Stability
    Analysis of Two-Relaxation-Times Lattice Boltzmann Schemes for
    Anisotropic Advection-Diffusion Equation", Commun. Comput. Phys.
    ABSTRACT ONLY.
  doi:10.1103/PhysRevE.95.013304. ABSTRACT ONLY: bounce-back no-flux on a
    staircase wall introduces spurious boundary-layer diffusion AND
    dispersion — invisible in a concentration profile, visible in moments;
    and the ADE equilibrium weights form TWO adjustable families setting the
    convective and diffusive stencils independently.
  Kruger, T. et al. (2017), "The Lattice Boltzmann Method", Springer.
    NOT READ — general-knowledge citation for the Chapman-Enskog framework;
    nothing downstream rests on it.
  He, X., Zou, Q., Luo, L.-S. & Dembo, M. (1997), J. Stat. Phys. 87:115-136.
    NOT READ. Still the anchor to check before any momentum wall-position
    claim is made here.
"""

import itertools

import numpy as np

# --------------------------------------------------------------------------
# Velocity sets. The weights are the input; every derived quantity below is
# computed from them rather than tabulated, so a wrong weight fails loudly.
# --------------------------------------------------------------------------


def _d3q19():
    c = [(0, 0, 0)] + [
        v
        for v in itertools.product((-1, 0, 1), repeat=3)
        if sum(abs(x) for x in v) in (1, 2)
    ]
    w = [1.0 / 3.0] + [
        1.0 / 18.0 if sum(abs(x) for x in v) == 1 else 1.0 / 36.0 for v in c[1:]
    ]
    return c, w


def _d3q27():
    c = list(itertools.product((-1, 0, 1), repeat=3))
    table = {0: 8.0 / 27.0, 1: 2.0 / 27.0, 2: 1.0 / 54.0, 3: 1.0 / 216.0}
    return c, [table[sum(abs(x) for x in v)] for v in c]


VELOCITY_SETS = {
    "D1Q3": (
        [(0,), (1,), (-1,)],
        [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0],
    ),
    "D2Q5": (
        [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)],
        [1.0 / 3.0] + [1.0 / 6.0] * 4,
    ),
    "D2Q9": (
        [(0, 0), (1, 0), (0, 1), (-1, 0), (0, -1),
         (1, 1), (-1, 1), (-1, -1), (1, -1)],
        [4.0 / 9.0] + [1.0 / 9.0] * 4 + [1.0 / 36.0] * 4,
    ),
    "D3Q7": (
        [(0, 0, 0), (1, 0, 0), (-1, 0, 0), (0, 1, 0),
         (0, -1, 0), (0, 0, 1), (0, 0, -1)],
        [1.0 / 4.0] + [1.0 / 8.0] * 6,
    ),
    "D3Q19": _d3q19(),
    "D3Q27": _d3q27(),
}

# c_s^2 FOR THE WEIGHTS TABULATED ABOVE — not for the lattice name. D3Q7 and
# D2Q5 are omega-families (see the docstring and the *_weights functions);
# these entries are their omega = 3/4 and omega = 2/3 members respectively.
SOUND_SPEED_SQUARED = {
    "D1Q3": 1.0 / 3.0,
    "D2Q5": 1.0 / 3.0,
    "D2Q9": 1.0 / 3.0,
    "D3Q7": 1.0 / 4.0,
    "D3Q19": 1.0 / 3.0,
    "D3Q27": 1.0 / 3.0,
}

# Sets whose fourth moment is isotropic, and which can therefore carry
# momentum. The others are scalar-only. Computed in `verify_limits`.
NAVIER_STOKES_CAPABLE = ("D2Q9", "D3Q19", "D3Q27")

UNRESOLVED = {
    "ma_squared_off_axis": (
        "The Ma^2 result D_eff = (c_s^2 - u^2)(tau - 1/2) is DERIVED and "
        "lattice-verified for flow along a lattice axis on D1Q3 and the "
        "D2Q5(omega) family (see diffusivity_first_order_eq). Diagonal flow "
        "directions and the 3-D sets are conjectured to follow the same law "
        "and are NOT derived; anisotropy of the error tensor is untested. "
        "SCOPE AT LARGE TAU: the law is the k -> 0 coefficient, and the "
        "k-expansion saturates slowly when tau is large. At tau = 5, "
        "u = 0.4 (OpenLB's shipped ADE benchmark), the k -> 0 law gives "
        "0.78 while the exact conserved eigenvalue at the benchmark's own "
        "k = 2 pi/51 gives 0.9031 -- and OpenLB's field data measures "
        "0.908 (tools/openlb_first_contact.py). Quote the eigenvalue at "
        "the actual wavenumber when tau is large, not the limit law."
    ),
    "bounce_back_wall_position": (
        "MOMENTUM lattice: for halfway bounce-back the effective wall "
        "location is reputedly tau-dependent, with anchors He, Zou, Luo & "
        "Dembo (1997) and Ginzburg's Lambda = 3/16. STILL OPEN: the one "
        "research claim tying this to a readable source failed adversarial "
        "verification 0-3, and the primary papers remain unread, so no "
        "coefficient is given. NOTE the SCALAR-lattice analogue is now "
        "RESOLVED and is different in kind: its slip converges as 1/N^2 "
        "(see ade_dirichlet_slip), so do not carry the 'refinement-"
        "independent' framing across to it."
    ),
    # RESOLVED 2026-08-11 against an OpenLB 1.9.0 checkout (gitlab.com/
    # openlb/release, commit 3dedbdd2c4d5). src/descriptor/definition/
    # common.h defines cs2<3,7> = {1,4} with weights t<3,7> = {1/4, 1/8 x6}
    # — the omega = 3/4 family member, exactly as inferred from guide
    # Eq. 4.58. The same file defines cs2<2,5> = {1,3} (weights 1/3, 1/6 x4,
    # the omega = 2/3 member) for the GENERAL D2Q5 descriptor, while the
    # guide's thermal MRT D2Q5 declares c_s^2 = 0.2 — so the two-conventions-
    # under-one-name trap exists WITHIN OpenLB itself, not only across the
    # literature. And src/dynamics/advectionDiffusionDynamics.h builds every
    # ADE dynamics from equilibria::FirstOrder (lines 107, 177, 189), which
    # upgrades the guide-level 2-1 claim to source-verified: the Ma^2
    # depletion law D_eff = (c_s^2 - u^2)(tau - 1/2) is the law OpenLB's
    # scalar lattice obeys.
}


def _check_set(name):
    if name not in VELOCITY_SETS:
        raise ValueError(
            f"velocity set must be one of {tuple(VELOCITY_SETS)}, got {name!r}"
        )
    return name


def sound_speed_squared(velocity_set):
    """c_s^2 for the named set. 1/3 everywhere except D3Q7, which is 1/4."""
    return SOUND_SPEED_SQUARED[_check_set(velocity_set)]


def diffusivity(tau, velocity_set="D3Q19", dt=1.0):
    """D = c_s^2 (tau - 1/2) dt.

    NEGATIVE for tau < 1/2, which is the method's stability boundary written
    as a transport coefficient. Returned rather than raised on, because a
    negative value IS the diagnostic: a code that reports one has been asked
    for a diffusivity it cannot represent.
    """
    return sound_speed_squared(velocity_set) * (float(tau) - 0.5) * float(dt)


def tau_from_diffusivity(d, velocity_set="D3Q19", dt=1.0):
    """Invert the relation: tau = D/(c_s^2 dt) + 1/2."""
    return float(d) / (sound_speed_squared(velocity_set) * float(dt)) + 0.5


def diffusivity_naive(tau, velocity_set="D3Q19", dt=1.0):
    """D = c_s^2 tau dt — the WRONG relation, with the -1/2 dropped.

    Provided so a case can quantify the error rather than describe it. This is
    not a strawman: it is the form that appears if the Chapman-Enskog
    discrete-time correction is omitted, and it is right to within 33% at
    tau = 2 while being wrong by 51x at tau = 0.51.
    """
    return sound_speed_squared(velocity_set) * float(tau) * float(dt)


def naive_error_factor(tau):
    """How many times too large the naive relation is: tau/(tau - 1/2).

    Independent of the velocity set and of dt, since c_s^2 dt cancels. Diverges
    as tau -> 1/2, which is the low-diffusivity limit a solute transport case
    lives in.
    """
    return float(tau) / (float(tau) - 0.5)


def mach_number(u, velocity_set="D3Q19"):
    """Ma = u / c_s, in lattice units. The O(Ma^2) error scales with its
    square; see UNRESOLVED for why no coefficient is given."""
    return float(u) / np.sqrt(sound_speed_squared(velocity_set))


def is_navier_stokes_capable(velocity_set):
    """Whether the set's FOURTH moment is isotropic.

    D2Q5 and D3Q7 fail this: they can carry an advection-diffusion equation
    but not momentum. Computed in `verify_limits` from the velocities.
    """
    return _check_set(velocity_set) in NAVIER_STOKES_CAPABLE


def moment_tensors(velocity_set):
    """(M0, M2, M4_xxxx, M4_xxyy) computed from the discrete velocities.

    M0 = sum w_i, M2_ab = sum w_i c_ia c_ib. These are the quantities the
    Chapman-Enskog closure needs, so computing them directly is what turns the
    tabulated constants above into derived ones.
    """
    _check_set(velocity_set)
    c, w = VELOCITY_SETS[velocity_set]
    c = np.asarray(c, dtype=float)
    w = np.asarray(w, dtype=float)
    m0 = float(w.sum())
    m2 = np.einsum("i,ia,ib->ab", w, c, c)
    m4_xxxx = float(np.einsum("i,i->", w, c[:, 0] ** 4))
    m4_xxyy = (
        float(np.einsum("i,i->", w, c[:, 0] ** 2 * c[:, 1] ** 2))
        if c.shape[1] > 1
        else None
    )
    return m0, m2, m4_xxxx, m4_xxyy


def d3q7_weights(omega):
    """The D3Q7 omega-family: w_0 = 1 - omega, w_1..6 = omega/6.

    c_s^2 = omega/3, so the 'lattice constant' is a free choice: omega = 3/4
    is the textbook set (c_s^2 = 1/4) and omega = 1 the rest-free one
    (c_s^2 = 1/3). Source: arXiv:1801.00504 Table 1, re-verified here.
    """
    if not 0.0 < omega <= 1.0:
        raise ValueError(f"omega must be in (0, 1], got {omega}")
    return [1.0 - omega] + [omega / 6.0] * 6


def d2q5_weights(omega):
    """The D2Q5 omega-family: w_0 = 1 - omega, w_1..4 = omega/4; c_s^2 = omega/2.

    omega = 2/3 gives the tabulated set (c_s^2 = 1/3); omega = 2/5 gives
    OpenLB's thermal MRT set (w_0 = 3/5, c_s^2 = 1/5 — the guide's declared
    0.2). Same lattice name, different constant, which is the trap.
    """
    if not 0.0 < omega <= 1.0:
        raise ValueError(f"omega must be in (0, 1], got {omega}")
    return [1.0 - omega] + [omega / 4.0] * 4


def sound_speed_squared_from_weights(velocities, weights):
    """c_s^2 = sum_i w_i |c_i|^2 / d — from the weights ACTUALLY IN USE.

    This, not the name-keyed table, is the interface an analytic reference should use
    against a real solver: the same lattice name carries different constants
    under different rest weights, and OpenLB itself ships two.
    """
    c = np.asarray(velocities, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(c) != len(w):
        raise ValueError("velocities and weights differ in length")
    return float(np.einsum("i,i->", w, np.einsum("ia,ia->i", c, c)) / c.shape[1])


# --------------------------------------------------------------------------
# The ADE Dirichlet wall slip (anti-bounce-back), and the magic parameter.
# arXiv:1603.09577 Eqs. 71-73; every identity re-verified in verify_limits.
# --------------------------------------------------------------------------


def magic_lambda(tau):
    """Ginzburg's magic parameter for BGK: Lambda = (tau - 1/2)^2."""
    return (float(tau) - 0.5) ** 2


def ade_dirichlet_slip(tau, n_cells, delta_phi=1.0):
    """Spurious uniform offset of anti-bounce-back Dirichlet walls, BGK.

        phi_s = (dphi / (12 N^2)) * 16 * [ (tau - 1/2)^2 - 3/16 ]

    (the published bracket 4(2/s - 1)^2 - 3 with s = 1/tau, identically).
    Zero exactly at Lambda = 3/16. SCALES AS 1/N^2: this inflates the
    second-order error constant rather than destroying the order, so it is
    NOT the wedge-bias analogue — a refinement study sees it shrink, and what
    refinement cannot reveal is that one tuning choice removes it entirely.
    """
    return (
        float(delta_phi)
        / (12.0 * float(n_cells) ** 2)
        * 16.0
        * (magic_lambda(tau) - 3.0 / 16.0)
    )


def zero_slip_tau():
    """tau at which the BGK anti-bounce-back slip vanishes: 1/2 + sqrt(3)/4.

    = 0.93301, i.e. s = 1/tau = 4(2 - sqrt(3)) = 1.07180. The unique BGK
    member of Lambda = 3/16.
    """
    return 0.5 + np.sqrt(3.0) / 4.0


def mrt_zero_slip_s1(s3):
    """The CORRECTED zero-slip relation for MRT: s1 = 8(s3 - 2)/(s3 - 8).

    Satisfies (1/s3 - 1/2)(1/s1 - 1/2) = 3/16 identically — the TRT magic
    condition. THE PUBLISHED EQ. (73) of arXiv:1603.09577 PRINTS THE
    DENOMINATOR WITH THE OPPOSITE SIGN, 8(s3 - 2)/(8 - s3), which returns a
    NEGATIVE relaxation rate for every admissible s3 < 8 (s3 = 1 gives
    -8/7). A framework implementing the printed form would demand an
    unphysical parameter; `verify_limits` demonstrates both facts.
    """
    s3 = float(s3)
    if not 0.0 < s3 < 2.0:
        raise ValueError(f"s3 must be in (0, 2) for stability, got {s3}")
    return 8.0 * (s3 - 2.0) / (s3 - 8.0)


# --------------------------------------------------------------------------
# The O(Ma^2) advection error — derived in this repo by von Neumann analysis
# of the exact amplification matrix (the ADE update with a linear equilibrium
# is linear in the distributions), verified on an actual lattice to 3.7e-9.
# --------------------------------------------------------------------------


def diffusivity_first_order_eq(tau, u, velocity_set="D3Q19", dt=1.0):
    """D_eff = (c_s^2 - u^2)(tau - 1/2) dt — the FIRST-order-equilibrium BGK
    scheme's actual diffusivity, exactly, for flow along a lattice axis.

    This is what OpenLB's ADE lattice (guide Eq. 4.21) realises: the scheme's
    advection depletes its own diffusivity by Ma^2 = u^2/c_s^2. At u = 0.3 on
    a c_s^2 = 1/3 lattice that is a 27% deficit, and the concentration field
    still looks entirely plausible. Derived for D1Q3 and D2Q5(omega); other
    sets conjectured (see UNRESOLVED["ma_squared_off_axis"]).
    """
    cs2 = sound_speed_squared(velocity_set)
    return (cs2 - float(u) ** 2) * (float(tau) - 0.5) * float(dt)


def ma2_relative_error(u, velocity_set="D3Q19"):
    """The exact fractional diffusivity error of the first-order-equilibrium
    scheme: -u^2/c_s^2 = -Ma^2. The coefficient is exactly -1."""
    return -float(u) ** 2 / sound_speed_squared(velocity_set)


def d2q5_second_order_residual(u, omega):
    """Relative diffusivity error REMAINING with the second-order equilibrium
    on the D2Q5(omega) family: -u^2 (3 omega - 2) / omega^2.

    Zero only at omega = 2/3, where c_s^2 takes its standard 1/3. At OpenLB's
    thermal weights (omega = 2/5) it is +5 u^2 — the standard second-order
    term OVERCORRECTS, because it is built for moments the reduced weights do
    not have. Lattice-verified at (omega, tau, u) = (2/5, 1, 0.2): predicted
    +20.0%, measured ratio 1.000000000.
    """
    if not 0.0 < omega <= 1.0:
        raise ValueError(f"omega must be in (0, 1], got {omega}")
    return -float(u) ** 2 * (3.0 * omega - 2.0) / omega**2


def advection_magic_lambda():
    """Lambda = 1/12: the value at which the D1Q3 first-order scheme's
    third-order dispersive error E3 = 2 u^3 (Lambda - 1/12) vanishes.

    A DIFFERENT value from the wall slip's 3/16 (`zero_slip_tau`), confirming
    by direct derivation the TRT result that no single Lambda kills both. The
    corresponding BGK tau is 1/2 + 1/(2 sqrt(3)) = 0.78868.
    """
    return 1.0 / 12.0


def _dispersion_numeric(tau, u, weights, velocities_1d, order2, cs2, k=1e-10):
    """ln(conserved eigenvalue) coefficients from the EXACT amplification
    matrix at 50-digit precision — the independent route the closed forms are
    checked against. Tiny k needs no Richardson: at k = 1e-10 the k^2 term
    sits 30 digits above the mpmath floor."""
    from mpmath import mp, matrix, exp as mexp, log as mlog, mpc

    mp.dps = 50
    n = len(weights)
    kk = mp.mpf(k)
    # THE WEIGHTS MUST BE EXACT AT WORKING PRECISION. Passed as doubles,
    # 2/3 + 1/6 + 1/6 = 1 - 5e-17, and that normalisation defect sits
    # directly in the conserved eigenvalue — at k = 1e-10 it swamps the
    # physical D k^2 ~ 2e-22 by five orders of magnitude and read as
    # d_eff ~ 9252 on the first run of this check. Renormalise in mpf and
    # derive c_s^2 from the weights so every internal identity holds at
    # 50 digits by construction; the caller's cs2 argument only names the
    # equilibrium convention and is superseded here.
    w_mp = [mp.mpf(wi) for wi in weights]
    total = sum(w_mp)
    w_mp = [wi / total for wi in w_mp]
    cs2_mp = sum(wi * ei**2 for wi, ei in zip(w_mp, velocities_1d))
    q = []
    for wi, ei in zip(w_mp, velocities_1d):
        c = 1 + ei * mp.mpf(u) / cs2_mp
        if order2:
            c += (ei**2 - cs2_mp) * mp.mpf(u) ** 2 / (2 * cs2_mp**2)
        q.append(wi * c)
    G = matrix(n, n)
    for i in range(n):
        ph = mexp(mpc(0, -1) * kk * velocities_1d[i])
        for j in range(n):
            G[i, j] = ph * (
                (1 - 1 / mp.mpf(tau)) * (1 if i == j else 0) + q[i] / mp.mpf(tau)
            )
    from mpmath import eig

    vals = eig(G, left=False, right=False)
    lam = max(vals, key=lambda z: abs(z))  # conserved mode: |lambda| -> 1
    lnl = mlog(lam)
    u_eff = float(-lnl.imag / kk)
    d_eff = float(-lnl.real / kk**2)
    return u_eff, d_eff


def verify_limits(rtol=1e-13):
    """Derive the lattice constants from the velocity sets and check them.

    Nothing here restates a tabulated value: c_s^2 comes from the second
    moment, the Navier-Stokes capability from the fourth, and the
    relaxation-diffusivity relation is checked through its own inverse and its
    limits.
    """
    errors = {}

    for name in VELOCITY_SETS:
        m0, m2, m4x, m4xy = moment_tensors(name)

        # 1. The weights are a probability distribution.
        errors[f"{name}_weights_sum"] = abs(m0 - 1.0)

        # 2. The second moment is isotropic, and its diagonal IS c_s^2. This
        #    is where D3Q7's 1/4 comes from, rather than from a table.
        diag = np.diag(m2)
        off = m2 - np.diag(diag)
        errors[f"{name}_second_moment_isotropic"] = float(np.max(np.abs(off)))
        errors[f"{name}_second_moment_uniform"] = float(
            np.max(np.abs(diag - diag[0]))
        )
        errors[f"{name}_sound_speed"] = abs(
            float(diag[0]) / SOUND_SPEED_SQUARED[name] - 1.0
        )

        # 3. The fourth moment decides whether the set can carry momentum.
        cs2 = SOUND_SPEED_SQUARED[name]
        ok_xxxx = abs(m4x - 3.0 * cs2**2) < 1e-13
        ok_xxyy = True if m4xy is None else abs(m4xy - cs2**2) < 1e-13
        derived_capable = ok_xxxx and ok_xxyy
        if name == "D1Q3":
            # One dimension has no cross term, so the classification is not
            # meaningful; excluded rather than silently passed.
            continue
        if derived_capable != is_navier_stokes_capable(name):
            raise AssertionError(
                f"{name}: fourth-moment isotropy computed as {derived_capable} "
                f"but NAVIER_STOKES_CAPABLE says {is_navier_stokes_capable(name)}"
            )

    # 4. D3Q7 really is the odd one out, and by exactly 4/3.
    errors["d3q7_vs_others"] = abs(
        (SOUND_SPEED_SQUARED["D3Q19"] / SOUND_SPEED_SQUARED["D3Q7"]) / (4.0 / 3.0)
        - 1.0
    )

    # 5. The relaxation relation, through its own inverse.
    for name in VELOCITY_SETS:
        for tau in (0.51, 0.8, 1.0, 2.0, 5.0):
            d = diffusivity(tau, name, dt=0.7)
            errors[f"{name}_tau_roundtrip_{tau}"] = abs(
                tau_from_diffusivity(d, name, dt=0.7) / tau - 1.0
            )

    # 6. The limits: D = 0 at tau = 1/2 exactly, and NEGATIVE below it.
    errors["zero_at_half"] = abs(diffusivity(0.5, "D3Q19"))
    if diffusivity(0.49, "D3Q19") >= 0.0:
        raise AssertionError(
            "tau < 1/2 must give a NEGATIVE diffusivity; that is the stability "
            "boundary expressed as a transport coefficient"
        )

    # 7. The naive relation's error factor is set- and dt-independent, and
    #    diverges at tau -> 1/2. Both are the point of the diagnostic.
    for tau in (0.51, 0.6, 1.0, 2.0):
        ratio = diffusivity_naive(tau, "D3Q7", dt=3.3) / diffusivity(
            tau, "D3Q7", dt=3.3
        )
        errors[f"naive_factor_tau{tau}"] = abs(ratio / naive_error_factor(tau) - 1.0)
        other = diffusivity_naive(tau, "D2Q9", dt=0.1) / diffusivity(
            tau, "D2Q9", dt=0.1
        )
        errors[f"naive_factor_set_independent_tau{tau}"] = abs(other / ratio - 1.0)
    # The headline numbers quoted in the docstring.
    for tau, expect in ((2.0, 4.0 / 3.0), (1.0, 2.0), (0.6, 6.0), (0.51, 51.0)):
        errors[f"naive_quoted_tau{tau}"] = abs(naive_error_factor(tau) / expect - 1.0)

    # 8. Mach number uses the set's own sound speed, so D3Q7 differs.
    errors["mach_d3q7_vs_d3q19"] = abs(
        mach_number(0.1, "D3Q7") / mach_number(0.1, "D3Q19") - np.sqrt(4.0 / 3.0)
    )

    # 9. The omega-families reproduce both published constants from the SAME
    #    formula, which is what dissolves the 1/4-vs-1/3 confusion.
    for omega, expect in ((0.75, 0.25), (1.0, 1.0 / 3.0)):
        w = d3q7_weights(omega)
        cs2 = sound_speed_squared_from_weights(VELOCITY_SETS["D3Q7"][0], w)
        errors[f"d3q7_family_omega{omega}"] = abs(cs2 / expect - 1.0)
    for omega, expect in ((2.0 / 3.0, 1.0 / 3.0), (0.4, 0.2)):
        w = d2q5_weights(omega)
        cs2 = sound_speed_squared_from_weights(VELOCITY_SETS["D2Q5"][0], w)
        errors[f"d2q5_family_omega{round(omega, 3)}"] = abs(cs2 / expect - 1.0)
    # ... and the tabulated sets are the omega = 3/4 and omega = 2/3 members.
    errors["d3q7_table_is_family_member"] = abs(
        sound_speed_squared_from_weights(*VELOCITY_SETS["D3Q7"])
        / SOUND_SPEED_SQUARED["D3Q7"]
        - 1.0
    )

    # 10. The ADE Dirichlet slip: the bracket identity, the zero, and its
    #     1/N^2 scaling — checked against the PUBLISHED form 4(2/s-1)^2 - 3
    #     rather than against this module's own rewriting of it.
    for tau in (0.6, 0.93301270189, 1.0, 2.0):
        s_rate = 1.0 / tau
        published = 4.0 * (2.0 / s_rate - 1.0) ** 2 - 3.0
        mine = 16.0 * (magic_lambda(tau) - 3.0 / 16.0)
        errors[f"slip_bracket_tau{tau}"] = abs(mine - published) / max(
            abs(published), 1e-30
        ) if abs(published) > 1e-9 else abs(mine - published)
    errors["slip_zero_at_magic"] = abs(
        ade_dirichlet_slip(zero_slip_tau(), 64)
    )
    errors["zero_slip_s_value"] = abs(
        1.0 / zero_slip_tau() / (4.0 * (2.0 - np.sqrt(3.0))) - 1.0
    )
    errors["magic_lambda_at_zero_slip"] = abs(
        magic_lambda(zero_slip_tau()) / (3.0 / 16.0) - 1.0
    )
    errors["slip_scales_inverse_n2"] = abs(
        ade_dirichlet_slip(1.0, 32) / ade_dirichlet_slip(1.0, 64) / 4.0 - 1.0
    )

    # 11. The MRT relation: corrected form gives Lambda = 3/16 identically
    #     and a POSITIVE rate; the published form gives a negative one.
    for s3 in (0.8, 1.0, 1.5):
        s1 = mrt_zero_slip_s1(s3)
        if s1 <= 0.0:
            raise AssertionError("corrected MRT relation returned s1 <= 0")
        errors[f"mrt_magic_s3_{s3}"] = abs(
            (1.0 / s3 - 0.5) * (1.0 / s1 - 0.5) / (3.0 / 16.0) - 1.0
        )
        published_s1 = 8.0 * (s3 - 2.0) / (8.0 - s3)
        if published_s1 >= 0.0:
            raise AssertionError(
                "the PUBLISHED Eq. (73) should give a negative s1; if it no "
                "longer does, the sign-error record is stale"
            )
    # BGK is the s1 = s3 diagonal of the MRT family: the magic product at
    # s1 = s3 = 1/zero_slip_tau() must again be 3/16.
    s_bgk = 1.0 / zero_slip_tau()
    errors["bgk_is_mrt_diagonal"] = abs(
        (1.0 / s_bgk - 0.5) ** 2 / (3.0 / 16.0) - 1.0
    )

    # 12. THE Ma^2 LAW, against the exact amplification matrix at 50-digit
    #     precision — an independent numeric route, not the symbolic
    #     derivation re-run. D1Q3 first- and second-order equilibria, and the
    #     D2Q5 family residual at OpenLB's thermal weights.
    d1q3_v = [0, 1, -1]
    d1q3_w = [2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0]
    for tau, u in ((0.6, 0.3), (1.0, 0.1), (2.0, 0.3)):
        u_eff, d_eff = _dispersion_numeric(tau, u, d1q3_w, d1q3_v, False, 1.0 / 3.0)
        errors[f"ma2_first_order_t{tau}_u{u}"] = abs(
            d_eff / ((1.0 / 3.0 - u**2) * (tau - 0.5)) - 1.0
        )
        errors[f"ma2_u_exact_t{tau}_u{u}"] = abs(u_eff / u - 1.0)
    _, d_eff = _dispersion_numeric(1.0, 0.3, d1q3_w, d1q3_v, True, 1.0 / 3.0)
    errors["ma2_second_order_cancels"] = abs(d_eff / ((1.0 / 3.0) * 0.5) - 1.0)
    om = 0.4
    d2q5_v = [0, 1, -1, 0, 0]
    d2q5_w = [1.0 - om] + [om / 4.0] * 4
    _, d_eff = _dispersion_numeric(1.0, 0.2, d2q5_w, d2q5_v, True, om / 2.0)
    expected = (om / 2.0) * 0.5 * (1.0 + d2q5_second_order_residual(0.2, om))
    errors["ma2_d2q5_family_residual"] = abs(d_eff / expected - 1.0)
    # ... and the residual really is zero at omega = 2/3, or the "standard
    # weights are special" claim is hollow.
    errors["ma2_d2q5_zero_at_two_thirds"] = abs(
        d2q5_second_order_residual(0.5, 2.0 / 3.0)
    )
    # The two magic values are DIFFERENT, which is the no-free-lunch result.
    if abs(advection_magic_lambda() - 3.0 / 16.0) < 0.05:
        raise AssertionError("advection and wall magic values should differ")

    # 13. The unresolved items are DECLARED, not silently absent.
    if not UNRESOLVED.get("bounce_back_wall_position"):
        raise AssertionError(
            "the tau-dependent wall-position gap must stay declared until it "
            "is measured; deleting the entry is not the same as closing it"
        )

    for name, err in errors.items():
        if not err < rtol:
            raise AssertionError(
                f"lattice_boltzmann analytic reference {name} error {err:.3e} > {rtol:.0e}"
            )
    return {k: float(v) for k, v in errors.items()}
