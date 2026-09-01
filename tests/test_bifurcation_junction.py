"""The Y-junction: the programme's physics rung, against its gates.

docs/bifurcation-preregistration.md defines everything this test checks:
the geometry (Murray daughters at ±30°), gates G1-G3/G5/G7 with stated
origins, P2 as a hard expectation, and P1/P4 as reported measurements
with named alternatives. There is NO closed-form junction model — the
referee roles pass to conservation identities, the mirror symmetry, the
same-instrument straight baseline (the G4 record's N0 leg), and the
straight-pipe analytic model evaluated at matched path distance as the
comparison curve the departures are quoted against.

THE G3 FINDING (measured during bring-up, res 6, kept per policy): the
solved field carries a small antisymmetric bulk mode seeded near the
branch — 0.19% of u_max under bounce-back walls, 0.5% under Bouzidi —
although the discrete cell sets are EXACTLY mirror-symmetric (verified
pairwise; outlet discs 70/70 cells). Alignment is excluded; the leading
candidates are the oblique outlet discs' boundary-normal classification
and wall-scheme interactions. The gates below bound its scalar-level
effect and the record carries the resolution scaling as it accumulates.

P3 (the crossover clock in the daughters, 0.87x) needs a receiver-
distance sweep within one daughter per the pre-registration and is NOT
extracted here; the tail metrics that sweep will consume are persisted.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
OPENLB_ROOT = Path.home() / "GitHub" / "openlb"
CASE_FILE = REPO / "betaflow" / "cases" / "mc_channel.yaml"
G4_RECORD = REPO / "results" / "bifurcation_g4_control.json"
RESULTS_FILE = REPO / "results" / "bifurcation_junction.json"

pytestmark = pytest.mark.slow


@pytest.mark.skipif(
    not (OPENLB_ROOT / "build" / "lib" / "libolbcore.a").is_file(),
    reason="OpenLB build tree not present")
def test_bifurcation_junction():
    case = yaml.safe_load(CASE_FILE.read_text())
    res = run_case(case, runner="openlb", junction=True)
    t = res["t"]
    m = res["mass_over_initial"]
    fl = res["meta"]["fluid_stage"]

    # G1 — scalar mass: parking only, bounded, never growth. Floor origin:
    # the bent NT legs settled at 0.877-0.883; the junction adds apex cut
    # cells, budgeted one notch looser.
    assert m[0] == pytest.approx(1.0)
    assert m[-1] > 0.82, f"bulk mass fell to {m[-1]:.4f}"
    assert m[-1] < 1.005, f"bulk mass GREW to {m[-1]:.4f}"

    # G2 — fluid flux balance at the pre-registered 5e-3.
    assert abs(fl["flux_balance"]) < 5e-3, (
        f"G2: Q_daughters/Q_mother - 1 = {fl['flux_balance']:+.4f}")
    # G3 (fluid level) — the split asymmetry, bounded as a bug-catch while
    # the bring-up finding above is tracked; the value is recorded.
    assert abs(fl["flux_split_asymmetry"]) < 2e-2, (
        f"G3: Q+/Q- - 1 = {fl['flux_split_asymmetry']:+.4f}")
    # G5/G7 — profiles redeveloped against the local parabola.
    assert fl["mother_L2_vs_parabola"] < 2e-2
    assert fl["daughter_plus_L2_vs_parabola"] < 5e-2
    assert fl["daughter_minus_L2_vs_parabola"] < 5e-2
    # P2 — no recirculation past the carina (hard, direction only; the
    # measured margin is recorded).
    u_min = float(fl["gates"]["u_min_daughter_centreline"])
    assert u_min > 0.0, (
        f"P2 REFUTED: reversed flow on a daughter centreline ({u_min}); "
        f"a finding — report it, never absorb it")

    # Window metrics: peak, lag against the straight-pipe peak time at
    # matched path distance, tail ratio against the straight-pipe model.
    metrics = {}
    for name, w in res["windows"].items():
        cm, ref, t2 = w["cir"], w["ref_straight"], w["t2"]
        rsel = (t >= 3.0 * t2) & (t <= 6.0 * t2)
        metrics[name] = {
            "dbar_um": w["dbar"] * 1e6,
            "peak": float(np.max(cm)),
            "t_peak_over_straight_t2": float(t[np.argmax(cm)] / t2),
            "tail_ratio_3_to_6_t2_vs_straight_model":
                float(np.mean(cm[rsel]) / np.mean(ref[rsel])),
            "min": float(np.min(cm)),
        }

    # G3 (scalar level) — mirror windows on the same deterministic solve.
    # Tolerance origin: the measured antisymmetric fluid mode (0.5% of
    # u_max at res 6) propagated through transport; an order above the
    # bring-up scalar deltas (1.5-2.4e-3), far below every physics number
    # quoted.
    mirror = {}
    for kk in ("750", "1550"):
        dp = res["windows"][f"daughter_plus_{kk}"]["cir"]
        dm = res["windows"][f"daughter_minus_{kk}"]["cir"]
        delta = float(np.max(np.abs(dp - dm)))
        mirror[kk] = delta
        assert delta < 1e-2, (
            f"G3 scalar: mirror windows at {kk} um differ by {delta:.4f}")

    # P1 — the peak split, REPORTED against the same-instrument straight
    # baseline (the G4 record's N0 leg): the naive picture is one half.
    g4 = json.loads(G4_RECORD.read_text())
    n0 = {r["dbar_um"]: r for r in g4["legs"]["N0"]["receivers"]}
    p1 = {}
    for kk, um in (("750", 750.0), ("1550", 1550.0)):
        straight_peak = n0[um]["peak_measured"]
        p1[kk] = {
            "daughter_peak": metrics[f"daughter_plus_{kk}"]["peak"],
            "straight_peak_same_instrument": straight_peak,
            "ratio_to_half_straight":
                metrics[f"daughter_plus_{kk}"]["peak"]
                / (0.5 * straight_peak),
        }

    record = {
        "case": "mc_channel through the pre-registered Y-junction",
        "preregistration": "docs/bifurcation-preregistration.md",
        "geometry": {"murray_radius_um": 200.0 * 2.0**(-1.0/3.0),
                     "half_angle_deg": 30.0,
                     "s_junction_um": 350.0},
        "fluid_stage": fl,
        "windows": metrics,
        "mirror_window_max_abs_delta": mirror,
        "P1_peak_split": {
            "values": p1,
            "reading": "quoted against half the same-instrument straight "
                       "peak (G4 N0); departures from 1.0 are junction "
                       "physics plus the stated wall-layer and readout "
                       "budgets — the named alternatives of the "
                       "pre-registration apply",
        },
        "P2_no_recirculation": {
            "u_min_daughter_centreline": u_min,
            "confirmed": u_min > 0.0,
        },
        "P3_status": "NOT extracted: needs the receiver-distance sweep "
                     "within one daughter per the pre-registration; the "
                     "tail metrics above are its inputs",
        "g3_bringup_finding": "antisymmetric bulk mode seeded near the "
                              "branch: 0.19% of u_max (bounce-back) / "
                              "0.5% (Bouzidi) at res 6 with EXACTLY "
                              "mirror cell sets - alignment excluded; "
                              "candidates: oblique outlet disc normal "
                              "classification, wall-scheme interaction",
        "mass_final_over_initial": float(m[-1]),
        "meta": {k: v for k, v in res["meta"].items()},
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")
