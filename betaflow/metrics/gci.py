"""Grid Convergence Index (ASME V&V 20) — for quantities with NO exact oracle.

Everywhere an oracle exists, betaflow measures the TRUE discretisation error
and runs an order-of-accuracy test, which is strictly stronger. GCI is used
only where no exact solution is available, and it ESTIMATES rather than
measures. Results that rest on it must say so.
"""

import math


def grid_convergence_index(
    coarse,
    medium,
    fine,
    r=2.0,
    safety_factor=1.25,
    order_band=(1.5, 2.5),
    fallback_safety_factor=3.0,
):
    """GCI on a scalar functional from three solutions at refinement ratio r.

    Parameters
    ----------
    coarse, medium, fine : float
        The functional on the three grids, coarsest first.
    r : float
        Refinement ratio between successive grids (constant).
    safety_factor : float
        Fs for a sequence in the asymptotic range (ASME: 1.25 for three grids).
    order_band : (float, float)
        Band in which the observed order is accepted as asymptotic. Outside
        it, `fallback_safety_factor` is used and `asymptotic` is False — the
        caller MUST report that rather than quoting a tight band on an
        unconverged sequence.

    Returns
    -------
    dict with the observed order p, the extrapolated value, GCI on the fine
    grid (a relative error BAND, not an error), the Fs actually used, and
    whether the sequence was in the asymptotic range.
    """
    eps_32 = medium - coarse
    eps_21 = fine - medium
    if eps_21 == 0.0:
        return {
            "p": None,
            "gci_fine": 0.0,
            "safety_factor": safety_factor,
            "asymptotic": True,
            "note": "solutions identical to machine precision",
        }
    ratio = eps_32 / eps_21
    # A negative ratio means the sequence is oscillatory, not monotone: the
    # order is not defined and the result is not asymptotic.
    monotone = ratio > 0.0
    p = math.log(abs(ratio)) / math.log(r) if abs(ratio) > 0 else None
    asymptotic = bool(monotone and p is not None and order_band[0] <= p <= order_band[1])
    fs = safety_factor if asymptotic else fallback_safety_factor
    if p is None or abs(r**p - 1.0) < 1e-12:
        gci = float("inf")
    else:
        gci = fs * abs(eps_21 / fine) / (r**p - 1.0)
    extrapolated = (
        fine + eps_21 / (r**p - 1.0) if p is not None and abs(r**p - 1.0) > 1e-12 else None
    )
    return {
        "p": p,
        "monotone": monotone,
        "extrapolated": extrapolated,
        "gci_fine": gci,
        "safety_factor": fs,
        "asymptotic": asymptotic,
        "note": (
            "asymptotic range: Fs = %.2f" % fs
            if asymptotic
            else "NOT in the asymptotic range (p=%s, monotone=%s) — Fs raised to %.1f"
            % (None if p is None else round(p, 3), monotone, fs)
        ),
    }
