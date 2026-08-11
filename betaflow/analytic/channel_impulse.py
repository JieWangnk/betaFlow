"""The molecular-communications channel impulse response (CIR), exact.

THE FIRST COMMS-FACING ORACLE. The molecular-communications field describes a
fluid channel the way radio engineers describe a wireless one: release a
pulse of particles at a transmitter, count the fraction inside a receiver
volume over time, and call that curve the channel impulse response. Every
downstream comms quantity (symbol timing, inter-symbol interference,
achievable data rate) is computed from it. This module gives the exact CIR
for the flow-dominated regime, which is the benchmark the anchor paper of
the LBM channel plan validates against.

THE SETUP (Hofmann et al. 2024, Fig. 1). A straight pipe of radius a with
steady Poiseuille flow, mean speed V. Particles are released at t = 0,
uniformly over the cross-section at x = 0. The receiver is a transparent
cylinder spanning the full radius, with axial extent c_x, centred a distance
dbar downstream. Diffusion is neglected (the flow-dominated regime), so each
particle simply rides its own streamline: x(t) = v(r) t.

THE DERIVATION IS TWO LINES, via a lemma worth naming. Under a uniform-area
release, the particle speed is UNIFORMLY distributed: with s = r/a carrying
the area measure 2s ds and v = 2V(1 - s^2), the fraction with v <= 2Vw is
exactly w. So at time t the particle positions are uniform on [0, 2Vt], and
the CIR is a window-overlap fraction:

    CIR(t) = |[dbar - c_x/2, dbar + c_x/2] ∩ [0, 2Vt]| / (2Vt)

which evaluates to the three branches of Hofmann Eq. (13):

    0                              for t <= t1
    1 - (dbar - c_x/2)/(2 V t)     for t1 <= t <= t2
    c_x / (2 V t)                  for t >= t2

with onset t1 = (dbar - c_x/2)/(2V), peak at t2 = (dbar + c_x/2)/(2V), and
peak value c_x/(dbar + c_x/2). All three branches, both times, and the peak
were re-derived here independently and confirmed by Monte Carlo before this
module was written; the published form is correct.

THE TAIL IS THE INTERESTING PART. CIR ~ c_x/(2Vt) decays only as 1/t,
because particles near the wall move arbitrarily slowly — the time-integral
of the tail grows as log(t) without bound. Real diffusion removes this, and
the MEASURED structure (runners/langevin.py CIR mode, recorded in
results/mc_channel_departure.json) has two regimes: the tail is first
ENHANCED above 1/t, because radial diffusion pumps the upstream reservoir
of still-slower particles into the window faster than it clears the window
population, and then TERMINATES — measured exactly zero by 12 t2 at the
middle receiver — with the crossover observed at ~ tau_r/beta_1^2, the
radial relaxation eigentime, not at tau_r itself. The tail is precisely
what sets inter-symbol interference, so past the crossover the
flow-dominated model is qualitatively wrong for ISI — it UNDERESTIMATES
interference while the tail is enhanced and OVERESTIMATES it after the
termination. The regime criterion (Hofmann Eqs. 9-10) is
Pe >> 4 dbar / a with Pe = V a / D, the same Peclet convention
`advection_diffusion.peclet` uses; at the paper's parameters Pe = 200
against 4 dbar/a = 3, 15, 31 for the three receiver distances, all
reproduced by `verify_limits`.

CITATIONS
  Hofmann, P., Zhou, P., Lee, C., Reisslein, M., Fitzek, F.H.P. &
    Chae, C.-B. (2024), "OpenFOAM Simulation of Microfluidic Molecular
    Communications: Method and Experimental Validation", IEEE Access,
    doi:10.1109/ACCESS.2024.3438243. READ (pp. 1-6 in session): Eqs. 6,
    9-10, 13-14 and Table 1 are the source of the setup and the parameter
    values; the closed form was verified against an independent derivation
    here rather than transcribed.
  Wicke et al. [Hofmann's ref 16] is the original source of the analytic
    model per Hofmann. NOT READ HERE — the attribution is carried via
    Hofmann and flagged as such. The formula does not rest on it: the
    derivation above is self-contained and `verify_limits` re-checks every
    branch by quadrature.
"""

import numpy as np

# Hofmann Table 1, the anchor benchmark's parameters (SI units).
HOFMANN_TABLE_1 = {
    "radius": 200e-6,
    "receiver_length": 100e-6,
    "mid_receiver_distances": (150e-6, 750e-6, 1550e-6),
    "peclet": 200.0,
}


def onset_time(v_mean, dbar, c_x):
    """t1 = (dbar - c_x/2)/(2V): first arrival, carried by the centreline
    particles moving at 2V."""
    return (dbar - c_x / 2.0) / (2.0 * v_mean)


def peak_time(v_mean, dbar, c_x):
    """t2 = (dbar + c_x/2)/(2V): the CIR maximum, when the fastest particles
    reach the receiver's far end."""
    return (dbar + c_x / 2.0) / (2.0 * v_mean)


def peak_value(dbar, c_x):
    """CIR(t2) = c_x/(dbar + c_x/2) — geometry only, independent of speed."""
    return c_x / (dbar + c_x / 2.0)


def cir(t, v_mean, dbar, c_x):
    """The flow-dominated channel impulse response, all three branches.

    Fraction of the released particles inside the receiver window at time t.
    Vectorised in t; exact for pure advection on Poiseuille flow with a
    uniform-area release.
    """
    tt = np.asarray(t, dtype=float)
    lo = dbar - c_x / 2.0
    hi = dbar + c_x / 2.0
    front = 2.0 * v_mean * tt
    with np.errstate(divide="ignore", invalid="ignore"):
        overlap = np.clip(np.minimum(hi, front) - lo, 0.0, None)
        out = np.where(front > 0.0, overlap / front, 0.0)
    return out


def tail_mass(t_from, t_to, v_mean, dbar, c_x):
    """Time-integral of the CIR over [t_from, t_to], t_from >= t2.

    Equals (c_x/(2V)) ln(t_to/t_from) — it DIVERGES logarithmically as
    t_to grows, which is the precise statement of why the flow-dominated
    model must fail in the far tail: wall particles never leave. This is the
    inter-symbol-interference proxy, and the quantity diffusion changes most.
    """
    t2 = peak_time(v_mean, dbar, c_x)
    if t_from < t2:
        raise ValueError(f"tail_mass needs t_from >= t2 = {t2}")
    return c_x / (2.0 * v_mean) * np.log(t_to / t_from)


def flow_dominated(peclet, dbar, radius):
    """Hofmann Eqs. (9)-(10): flow-dominated when Pe >> 4 dbar / a.

    Returns the ratio Pe / (4 dbar / a); >> 1 means flow-dominated, << 1
    dispersion-dominated. The paper's cases give 66, 13, 6.5.
    """
    return peclet / (4.0 * dbar / radius)


def verify_limits(rtol=1e-9):
    """Re-derive every claim by an independent route before use as truth."""
    errors = {}
    v_mean, dbar, c_x = 1.5e-3, 750e-6, 100e-6
    a = 200e-6

    # 1. The lemma: under the area measure 2s ds, P(v <= 2Vw) = w exactly.
    s = np.linspace(0.0, 1.0, 2_000_001)
    for w in (0.1, 0.37, 0.5, 0.9):
        inside = (1.0 - s**2) <= w
        cdf = float(np.trapezoid(2.0 * s * inside, s))
        errors[f"lemma_w{w}"] = abs(cdf - w)

    # 2. Every branch of the CIR against direct quadrature over the release
    #    distribution — the closed form is never compared with itself.
    t1 = onset_time(v_mean, dbar, c_x)
    t2 = peak_time(v_mean, dbar, c_x)
    for label, t in (("before", 0.5 * t1), ("rising", 0.5 * (t1 + t2)),
                     ("peak", t2), ("tail", 3.0 * t2), ("far_tail", 30.0 * t2)):
        x = 2.0 * v_mean * (1.0 - s**2) * t
        inside = (x >= dbar - c_x / 2.0) & (x <= dbar + c_x / 2.0)
        quad = float(np.trapezoid(2.0 * s * inside, s))
        closed = float(cir(t, v_mean, dbar, c_x))
        errors[f"branch_{label}"] = abs(quad - closed)

    # 3. Continuity at both branch points, and the peak value formula.
    # The CIR is continuous with a slope KINK at t1 and t2, so the two-sided
    # difference over a relative offset eps is ~ eps * |t * slope| = O(eps).
    # eps is chosen three decades under the limit so the check measures
    # continuity, not its own probe width.
    eps = 1e-12
    errors["continuity_t1"] = abs(
        float(cir(t1 * (1 + eps), v_mean, dbar, c_x))
        - float(cir(t1 * (1 - eps), v_mean, dbar, c_x))
    )
    errors["continuity_t2"] = abs(
        float(cir(t2 * (1 + eps), v_mean, dbar, c_x))
        - float(cir(t2 * (1 - eps), v_mean, dbar, c_x))
    )
    errors["peak_value"] = abs(
        float(cir(t2, v_mean, dbar, c_x)) / peak_value(dbar, c_x) - 1.0
    )
    # The peak is the maximum. The maximum sits at the KINK t2, which a
    # uniform grid misses by one step, so t2 is included as an exact sample
    # point; the check is then that the attained maximum IS the peak value
    # and no sampled time exceeds it.
    tt = np.append(np.linspace(0.5 * t1, 10.0 * t2, 200_001), t2)
    errors["peak_is_max"] = abs(
        float(np.max(cir(tt, v_mean, dbar, c_x))) / peak_value(dbar, c_x) - 1.0
    )

    # 4. The log-divergent tail: integral over [t2, T] = (cx/2V) ln(T/t2).
    T = 50.0 * t2
    tt = np.linspace(t2, T, 4_000_001)
    quad = float(np.trapezoid(cir(tt, v_mean, dbar, c_x), tt))
    errors["tail_log_integral"] = abs(
        quad / tail_mass(t2, T, v_mean, dbar, c_x) - 1.0
    )

    # 5. The paper's regime numbers reproduce, with the shared Pe convention.
    from betaflow.analytic import advection_diffusion as ad

    D = v_mean * a / HOFMANN_TABLE_1["peclet"]
    errors["peclet_convention"] = abs(
        ad.peclet(v_mean, a, D) / HOFMANN_TABLE_1["peclet"] - 1.0
    )
    for dbar_i, expect in zip(HOFMANN_TABLE_1["mid_receiver_distances"],
                              (3.0, 15.0, 31.0)):
        errors[f"regime_4dbar_over_a_{int(dbar_i*1e6)}"] = abs(
            4.0 * dbar_i / HOFMANN_TABLE_1["radius"] - expect
        )

    for name, err in errors.items():
        # Checks that integrate an INDICATOR function (the lemma and the
        # branch quadratures) inherit trapezoid's first-order error at the
        # jump: ~5e-7 at 2M points. That is the check's own limit, stated
        # here rather than absorbed silently into a global tolerance.
        limit = rtol
        if "branch" in name or "tail" in name or "lemma" in name:
            limit = 1e-6
        if not err < limit:
            raise AssertionError(
                f"channel_impulse oracle {name} error {err:.3e} > {limit:.0e}"
            )
    return {k: float(v) for k, v in errors.items()}
