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

  2. THE VELOCITY SETS, AND THE D3Q7 TRAP.

     c_s^2 = 1/3 for D1Q3, D2Q5, D2Q9, D3Q19 and D3Q27 — but c_s^2 = 1/4 for
     D3Q7, which is the smallest three-dimensional set and therefore the one a
     scalar lattice is most likely to use for speed. Substituting 1/3 there
     gives a diffusivity 4/3 too large at every tau, and nothing about the
     resulting field looks wrong.

     COMPUTED, not quoted: `verify_limits` forms sum_i w_i c_ia c_ib directly
     from each set's discrete velocities and reads c_s^2 off the diagonal.

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

WHAT THIS MODULE DELIBERATELY DOES NOT CLAIM. The O(Ma^2) advection error and
the tau-dependent effective wall position of bounce-back are both real and
both matter — the second is the exact analogue of the wedge faceting bias this
repo documents, being refinement-independent and invisible in a velocity
profile. Neither has a coefficient stated here, because neither has been
verified in this session. They are named in `UNRESOLVED` so the gap is
explicit rather than absent.

CITATIONS
  Kruger, T., Kusumaatmaja, H., Kuzmin, A., Shardt, O., Silva, G. & Viggen,
    E.M. (2017), "The Lattice Boltzmann Method: Principles and Practice",
    Springer. The standard reference for the Chapman-Enskog expansion, the
    velocity sets and the relaxation-transport relations.
    NOT READ IN THIS SESSION — cited from general knowledge and flagged as
    such. The relation D = c_s^2(tau - 1/2) dt is standard and appears in
    every LBM text; the ALGEBRAIC CONSEQUENCES asserted here are verified
    independently in `verify_limits`, so nothing downstream rests on the
    citation being located to a page.
  He, X., Zou, Q., Luo, L.-S. & Dembo, M. (1997), J. Stat. Phys. 87:115-136 —
    the analysis of bounce-back showing the effective wall location depends on
    the relaxation time. NOT READ. Recorded as the anchor to check before any
    wall-position claim is made here.
  Ginzburg, I. & d'Humieres, D. — two-relaxation-time schemes and the "magic
    parameter" Lambda = 3/16 that makes the bounce-back wall location
    tau-independent. NOT READ. Same status.
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

# c_s^2 by set. THE D3Q7 ENTRY IS THE TRAP: every other set is 1/3.
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
    "ma_squared_advection_error": (
        "LBM advection carries an O(Ma^2) error, Ma = u/c_s. NO coefficient is "
        "stated here because none has been verified in this session. The "
        "structure is usable without it: the error vanishes as u -> 0 and "
        "grows quadratically, so a velocity sweep at fixed tau, mesh and "
        "diffusivity isolates it — the same 'sweep, not comparison' logic that "
        "separates numerical from physical diffusion."
    ),
    "bounce_back_wall_position": (
        "For halfway bounce-back the effective wall sits at the link midpoint "
        "only for particular tau; otherwise its location is TAU-DEPENDENT. "
        "This is the exact analogue of the wedge faceting bias documented in "
        "this repo: refinement-independent, and invisible in a velocity "
        "profile. The published anchors are He, Zou, Luo & Dembo (1997) and "
        "Ginzburg's TRT 'magic parameter' Lambda = 3/16; NEITHER HAS BEEN READ "
        "HERE, so no coefficient is given. Establishing it is the highest-value "
        "open item for a lattice-Boltzmann vascular model."
    ),
    "ade_wall_condition": (
        "Anti-bounce-back for a Dirichlet concentration and bounce-back for "
        "zero flux. Whether the SCALAR lattice carries a tau-dependent "
        "placement error analogous to the momentum one is not established here."
    ),
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

    # 9. The unresolved items are DECLARED, not silently absent.
    if not UNRESOLVED.get("bounce_back_wall_position"):
        raise AssertionError(
            "the tau-dependent wall-position gap must stay declared until it "
            "is measured; deleting the entry is not the same as closing it"
        )

    for name, err in errors.items():
        if not err < rtol:
            raise AssertionError(
                f"lattice_boltzmann oracle {name} error {err:.3e} > {rtol:.0e}"
            )
    return {k: float(v) for k, v in errors.items()}
