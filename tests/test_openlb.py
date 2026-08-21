"""OpenLB leg of mc_channel — Tier-1 exam through runners/openlb.py.

Two tests. The default-tier one exercises the parameter-selection law
(pure arithmetic, no OpenLB install). The slow one builds and runs the
D3Q7 app at the committed configuration (deterministic — no RNG anywhere,
so the bands below are re-measurements, not statistics) and writes
results/mc_channel_openlb.json.

What the slow test asserts, and why the bands are where they are:
  - The unit-converter round trip: OpenLB's own AdeUnitConverter must
    realise the tau the stability law requested (checked inside the
    runner, hard failure).
  - Mass: bulk-only accounting declines as mass parks in bounce-back
    cells over a growing wetted footprint (mechanism in the app source);
    measured -2.7% at half-horizon on the scoping run. Band: decline
    only, bounded at 6%.
  - Timing: the measured peaks lag the flow-dominated t2 by ~4% — the
    physical-D departure the particle legs also measure (peak depression
    and lag), plus scheme dispersion. Band: late by 0-10%, never early.
  - The tail sits ABOVE the flow-dominated reference at the far receiver
    (same direction as the particle legs' enhancement regime); how much
    of that excess is numerical is Tier-2's question, answered by the
    benchmark collation against the Langevin leg, not asserted here.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.provenance import git_sha

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "mc_channel.yaml"
RESULTS_FILE = REPO / "results" / "mc_channel_openlb.json"
OPENLB_ROOT = Path.home() / "GitHub" / "openlb"

RESOLUTION = 12
U_LAT_TARGET = 0.04

# Momentum-rung bands, set from the recorded first sweep
# (results/openlb_wall_position.json; tools/openlb_wall_position_sweep.py
# owns the expensive N = 81 and tau-probe points). Measured at N = 21/41:
# bb shift -0.408/-0.314 dx, order 1.42; bouzidi shift -0.050/-0.025 dx,
# order 2.14.
BB_SHIFT_BAND = (-0.50, -0.20)
BZ_SHIFT_MAX = 0.12
BB_ORDER_BAND = (1.1, 1.7)
BZ_ORDER_BAND = (1.7, 2.6)


def _load():
    return yaml.safe_load(CASE_FILE.read_text())


def test_openlb_parameter_selection_law():
    """The scoping law, no OpenLB needed: stability pins tau against 1/2."""
    from betaflow.runners.openlb import scope_parameters

    case = _load()
    tau, dt, pred = scope_parameters(case, RESOLUTION, U_LAT_TARGET)
    # u_lat = (tau - 1/2) c_s^2 Pe_cell inverts exactly.
    assert pred["u_lat_centreline"] == pytest.approx(
        (tau - 0.5) * 0.25 * pred["peclet_cell"])
    # The point the law forces IS the sharp corner: the naive D formula is
    # wrong by an O(100) factor here, and only the eigenvalue law survives.
    assert 50 < pred["naive_over_exact_D_factor"] < 200
    # The predicted first-order depletion at this u_lat is sub-percent —
    # stated before any run so the benchmark can cite it.
    assert pred["depletion_relative_centreline"] == pytest.approx(
        U_LAT_TARGET**2 / 0.25)
    assert pred["depletion_relative_centreline"] < 0.01


@pytest.mark.slow
@pytest.mark.skipif(
    not (OPENLB_ROOT / "build" / "lib" / "libolbcore.a").is_file(),
    reason="OpenLB 1.9 build not present at ~/GitHub/openlb")
def test_mc_channel_openlb():
    from betaflow.runners import run_case

    case = _load()
    res = run_case(case, runner="openlb", resolution=RESOLUTION,
                   u_lat_target=U_LAT_TARGET, time_horizon_over_t2=6.5,
                   outputs=400)

    # Mass gate: decline only (parking), bounded.
    m = res["mass_over_initial"]
    assert m[0] == pytest.approx(1.0)
    assert m[-1] > 0.94, f"bulk mass fell to {m[-1]:.4f} of initial"
    assert m[-1] < 1.005, f"bulk mass GREW to {m[-1]:.4f} — a leak inward"

    receivers = []
    for rec in res["receivers"]:
        t, cm, co = rec["t"], rec["cir_measured"], rec["cir_reference"]
        t_peak = float(t[np.argmax(cm)])
        lag = t_peak / rec["t2"] - 1.0
        assert -0.005 < lag < 0.10, (
            f"dbar={rec['dbar']*1e6:.0f}um: peak at {lag:+.1%} of t2 — "
            f"the departure direction is LATE (physical D + dispersion), "
            f"never early")
        peak_ratio = float(np.max(cm)) / float(np.max(co))
        assert 0.85 < peak_ratio < 1.25, (
            f"dbar={rec['dbar']*1e6:.0f}um: peak {peak_ratio:.3f} of the "
            f"slug-averaged reference")
        ringing_min = float(np.min(cm))
        receivers.append({
            "dbar_um": rec["dbar"] * 1e6,
            "peak_measured": float(np.max(cm)),
            "peak_reference_slug_averaged": float(np.max(co)),
            "peak_lag_relative": lag,
            "ringing_min": ringing_min,
        })

    # Far receiver, late tail: ABOVE the flow-dominated reference — the
    # same direction the particle legs measured. The split between the
    # physical enhancement and numerical dispersion is Tier-2's question.
    far = next(r for r in res["receivers"] if abs(r["dbar"] - 1550e-6) < 1e-9)
    t, cm, co = far["t"], far["cir_measured"], far["cir_reference"]
    sel = (t >= 3.0 * far["t2"]) & (t <= 6.0 * far["t2"])
    assert float(np.mean(cm[sel] - co[sel])) > 0.0, (
        "far-receiver tail fell below the flow-dominated reference — "
        "opposite to the measured physical departure direction")

    record = {
        "case": "mc_channel",
        "solver_leg": "openlb (D3Q7 ADE, first-order equilibrium)",
        "receivers": receivers,
        "mass_final_over_initial": float(m[-1]),
        "far_tail_mean_excess_3_to_6_t2": float(np.mean(cm[sel] - co[sel])),
        "meta": res["meta"],
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")


@pytest.mark.slow
@pytest.mark.skipif(
    not (OPENLB_ROOT / "build" / "lib" / "libolbcore.a").is_file(),
    reason="OpenLB 1.9 build not present at ~/GitHub/openlb")
def test_pipe_momentum_openlb():
    """The MOMENTUM rung: OpenLB's fluid solver takes the pipe exam.

    Same case, exact solution, and metric as the OpenFOAM leg
    (pipe_poiseuille_steady.yaml), through openlb_cases/pipeFlow3d:
    steady forced Poiseuille, D3Q19 ForcedBGK, at FIXED tau = 0.53 across
    resolutions (bounce-back's wall position is tau-dependent, so a
    varying tau would conflate the two scalings).

    What is measured, per resolution and per wall treatment:
      - the profile L2 against the exact parabola for a wall AT r = a;
      - the effective radius a_eff from a parabola fit on the interior
        (|y| <= 0.85 a), whose shift (a_eff - a)/dx IS the wall position —
        the item declared UNRESOLVED in analytic/lattice_boltzmann.py
        since the LBM reference was written.

    Pre-registered predictions and their measured outcomes (the sign
    prediction was WRONG and is kept, per the correction policy):
      H-bb predicted the wall OUTSIDE the last fluid node at a fixed
      fraction of dx; MEASURED: the effective wall sits INSIDE the
      geometric radius, and the offset DECAYS as a_eff - a ~ dx^1.4
      rather than holding a fixed c dx (shift -0.41/-0.31/-0.24 dx at
      N = 21/41/81, full sweep in results/openlb_wall_position.json,
      tau-dependence 3% over the stability-allowed range — resolution
      dominates tau here). H-bouzidi CONFIRMED: shift ~ dx^2
      (-0.050/-0.025/-0.011 dx) and order 2.1 — the control validates
      the instrument and excludes the shared suspects (Ma^2, convergence
      budget), because both walls carry them identically.

    The bands below are re-measurements of a deterministic app, set from
    the recorded first sweep; results/pipe_openlb.json carries the runs.
    """
    import math

    from betaflow.analytic import pipe
    from betaflow.metrics import METRICS
    from betaflow.runners import run_case

    case = yaml.safe_load(
        (REPO / "betaflow" / "cases" / "pipe_poiseuille_steady.yaml")
        .read_text())
    a = float(case["geometry"]["radius"])
    out_file = REPO / "results" / "pipe_openlb.json"

    def fit_a_eff(y, u):
        sel = np.abs(y) <= 0.85 * a
        A = np.column_stack([np.ones(int(sel.sum())), y[sel] ** 2])
        coef, *_ = np.linalg.lstsq(A, u[sel], rcond=None)
        return float(np.sqrt(-coef[0] / coef[1])), float(coef[0])

    runs = []
    for wall in ("bb", "bouzidi"):
        for n_res in (21, 41):
            r = run_case(case, runner="openlb", resolution=n_res, tau=0.53,
                         wall=wall, workdir=REPO / "_runs")
            y, u = np.asarray(r["y"]), np.asarray(r["u"])
            l2 = METRICS["L2_velocity"](u / r["u_ref"],
                                        pipe.poiseuille_profile(y / a))
            a_eff, u_max_fit = fit_a_eff(y, u)
            dx = r["meta"]["dx"]
            runs.append({
                "wall": wall,
                "resolution": n_res,
                "tau": r["meta"]["tau_realised"],
                "L2_velocity": float(l2),
                "u_max_fit": u_max_fit,
                "a_eff": a_eff,
                "wall_shift_dx": (a_eff - a) / dx,
                "u_lat_char": r["meta"]["u_lat_char"],
            })
            # The Re definition-agreement check, same as the OpenFOAM leg.
            np.testing.assert_allclose(
                pipe.reynolds(r["meta"]["u_mean"], a, r["meta"]["nu"]),
                case["nondim"]["Re"], rtol=1e-12,
                err_msg="openlb runner and reference disagree on pipe Re")

    def order(rows):
        es = [x["L2_velocity"] for x in rows]
        ns = [x["resolution"] for x in rows]
        return [math.log(ec / ef) / math.log(nf / nc)
                for (ec, nc), (ef, nf) in zip(zip(es, ns), zip(es[1:], ns[1:]))]

    bb = [x for x in runs if x["wall"] == "bb"]
    bz = [x for x in runs if x["wall"] == "bouzidi"]
    record = {
        "case": "pipe_poiseuille_steady",
        "solver_leg": "openlb (D3Q19 ForcedBGK, openlb_cases/pipeFlow3d)",
        "fixed_tau": 0.53,
        "runs": runs,
        "observed_order": {"bb": order(bb), "bouzidi": order(bz)},
        "prediction_correction": (
            "H-bb's sign was wrong: predicted the wall outside the last "
            "fluid node, measured INSIDE the geometric radius. Kept per "
            "the correction policy."),
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_file.write_text(json.dumps(record, indent=2) + "\n")

    # --- demands, bands measured from the first sweep ---------------------
    for x in bb:
        assert BB_SHIFT_BAND[0] < x["wall_shift_dx"] < BB_SHIFT_BAND[1], (
            f"bb wall shift {x['wall_shift_dx']:+.3f} dx at N={x['resolution']} "
            f"outside {BB_SHIFT_BAND}")
    for x in bz:
        assert abs(x["wall_shift_dx"]) < BZ_SHIFT_MAX, (
            f"bouzidi wall shift {x['wall_shift_dx']:+.3f} dx at "
            f"N={x['resolution']} above {BZ_SHIFT_MAX}")
        # The control must beat the staircase decisively at every level.
        mate = next(b for b in bb if b["resolution"] == x["resolution"])
        assert x["L2_velocity"] < 0.5 * mate["L2_velocity"]
    for p_val in record["observed_order"]["bb"]:
        assert BB_ORDER_BAND[0] < p_val < BB_ORDER_BAND[1], (
            f"bb order {p_val:.2f} outside {BB_ORDER_BAND}")
    for p_val in record["observed_order"]["bouzidi"]:
        assert BZ_ORDER_BAND[0] < p_val < BZ_ORDER_BAND[1], (
            f"bouzidi order {p_val:.2f} outside {BZ_ORDER_BAND}")
    # The measured law: the shift MAGNITUDE shrinks under refinement
    # (a_eff - a ~ dx^1.4), so a fixed-c model would fail here.
    assert abs(bb[1]["wall_shift_dx"]) < abs(bb[0]["wall_shift_dx"])
