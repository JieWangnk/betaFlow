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
