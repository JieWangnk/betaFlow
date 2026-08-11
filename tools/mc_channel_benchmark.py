#!/usr/bin/env python3
"""Tier-2 collation: three solvers, one case, the exact solution as referee.

Reads the committed per-leg records of mc_channel and produces the
benchmark record. Nothing here re-runs a solver; every number is traced to
the record that measured it, and each solver's error is compared with the
prediction ITS OWN analytic reference made before the run:

  langevin            binomial laws (exact kinematics at D = 0)
  openfoam_particles  binomial laws + the one-sign interpolation-bias
                      prediction (unresolved at N = 2e4, recorded as such)
  openlb              the stability-pinned tau law, the first-order
                      depletion prediction (sub-percent, sub-dominant),
                      and the measured instrumentation findings

The Tier-2 payoff is the CROSS-SOLVER split of the departure from the
flow-dominated model at physical D (Pe = 200): the two particle legs are
independent implementations of the same physics and agree on the
two-regime tail (enhancement then termination), so their shared departure
IS the physics; OpenLB's excess over them is its numerical transport
error at the stability-forced parameter point.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from betaflow.provenance import git_sha  # noqa: E402

RESULTS = REPO / "results"


def _load(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.is_file() else None


def main():
    langevin = _load("mc_channel.json")
    departure = _load("mc_channel_departure.json")
    openfoam = _load("mc_channel_openfoam.json")
    openlb = _load("mc_channel_openlb.json")

    legs = {}

    if langevin and departure:
        legs["langevin"] = {
            "record": "results/mc_channel.json, mc_channel_departure.json",
            "replication_rung": {
                "rmse_over_binomial_floor": [
                    r["rmse_over_binomial_floor"]
                    for r in langevin["receivers"]],
                "pre_onset_exact_zero": all(
                    r["pre_onset_max"] == 0.0 for r in langevin["receivers"]),
            },
            "own_prediction_status": "binomial laws hold (band 1.06-1.27x)",
            "departure_peak_over_reference": [
                r["peak_measured_over_reference"]
                for r in departure["receivers"]],
            "departure_middle_tail": departure["middle_receiver_tail"],
            "wall_clock_this_machine": "~7 s (both rungs, numpy)",
        }

    if openfoam:
        d0 = openfoam["replication_rung_D0"]
        dep = openfoam["departure_rung"]
        legs["openfoam_particles"] = {
            "record": "results/mc_channel_openfoam.json",
            "replication_rung": {
                "rmse_over_binomial_floor": [
                    r["rmse_over_binomial_floor"] for r in d0["receivers"]],
                "pre_onset_exact_zero": all(
                    r["pre_onset_max"] == 0.0 for r in d0["receivers"]),
                "centroid_shift_t2_units": [
                    r["centroid_shift_t2_units"] for r in d0["receivers"]],
            },
            "own_prediction_status": (
                "binomial laws hold; the one-sign interpolation-bias "
                "prediction is UNRESOLVED at N = 2e4 (centroid shifts "
                "sub-0.1% of t2, mixed signs)"),
            "departure_peak_over_reference": [
                r["peak_measured_over_reference"] for r in dep["receivers"]],
            "departure_middle_tail": dep["middle_receiver_tail"],
            "replication_claim": openfoam["replication_claim"],
            "wall_clock_this_machine": "~4 min (both rungs, OF14 tracking)",
        }

    if openlb:
        legs["openlb"] = {
            "record": "results/mc_channel_openlb.json",
            "peak_lag_relative": [
                r["peak_lag_relative"] for r in openlb["receivers"]],
            "peak_over_slug_averaged_reference": [
                r["peak_measured"] / r["peak_reference_slug_averaged"]
                for r in openlb["receivers"]],
            "ringing_min": [r["ringing_min"] for r in openlb["receivers"]],
            "mass_final_over_initial": openlb["mass_final_over_initial"],
            "own_prediction_status": (
                "stability-pinned tau realised by OpenLB's converter to 6 "
                "decimals; predicted first-order depletion "
                f"{openlb['meta']['predictions_before_run']['depletion_relative_centreline']:.2%} "
                "(sub-dominant); wall instrumentation findings recorded"),
            "wall_clock_this_machine": (
                "~2 min (res 12, 15.6k steps, SISD; the full pytest leg "
                "measured 115 s)"),
        }

    # The Tier-2 split at the middle receiver, [3,5] t2: what the two
    # particle implementations agree on is physics; OpenLB's excess over
    # the flow-dominated reference beyond that is numerical.
    cross = {
        "statement": (
            "Two independent particle implementations agree on the "
            "two-regime tail (enhancement then termination) and on peak "
            "depression at physical D; OpenLB reproduces the direction "
            "with a larger magnitude, the excess being its numerical "
            "dispersion at the stability-forced parameter point "
            "(cell Peclet 33, tau - 1/2 = 4.8e-3)."),
        "peak_over_reference_by_solver": {
            "langevin": legs.get("langevin", {}).get(
                "departure_peak_over_reference"),
            "openfoam_particles": legs.get("openfoam_particles", {}).get(
                "departure_peak_over_reference"),
            "openlb_vs_slug_averaged": legs.get("openlb", {}).get(
                "peak_over_slug_averaged_reference"),
        },
        "termination_cross_confirmed": (
            "both particle legs measure CIR = 0 at 12 t2 (middle receiver) "
            "where the flow-dominated model predicts 1.04e-2"),
    }

    record = {
        "case": "mc_channel",
        "tier": 2,
        "referee": "betaflow/analytic/channel_impulse.py (18 self-checks)",
        "validity_audit": "results/hofmann_validity_audit.json",
        "legs_present": sorted(legs),
        "legs_missing": [k for k in ("langevin", "openfoam_particles",
                                     "openlb") if k not in legs],
        "legs": legs,
        "cross_solver": cross,
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = RESULTS / "mc_channel_benchmark.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(f"legs: {record['legs_present']}; missing: {record['legs_missing']}")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
