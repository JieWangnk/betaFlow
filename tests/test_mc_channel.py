"""mc_channel — the molecular-communications CIR against its exact analytic reference.

Two legs, one release each:

  EXACT KINEMATICS (D = 0). x(t) = u(r0) t in closed form, so the ONLY
  error is the finite-N release draw and every tolerance is a binomial law
  (metrics/mc_error.py). Pre-onset counts must be EXACTLY zero — not small,
  zero — because no particle moves faster than the centreline.

  DEPARTURE (D > 0, Pe = 200). The flow-dominated analytic reference fails in a
  MEASURED structure: softened onset, depressed peak, and a two-regime
  tail — enhanced while the upstream reservoir of slower particles feeds
  the window, then terminated (crossover clock: the layer-escape scaling
  t_cross = K tau_r^0.31 dbar^0.73, measured by the pre-registered Pe
  sweep; an earlier eigentime attribution is withdrawn there).
  The pre-measurement prediction was a DEPLETED tail; it
  was wrong in direction at intermediate times, and the assertions below
  encode what was measured, not what was expected.

The gate for both legs is the radial invariant P(r) = 2r/a^2: at D = 0 it
checks the release sampler, at D > 0 the specular wall scheme, whose bias
would enter the CIR invisibly through the velocity sampling.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.analytic import channel_impulse as ci
from betaflow.metrics import METRICS, mc_error
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "mc_channel.yaml"
RESULTS_FILE = REPO / "results" / "mc_channel.json"
DEPARTURE_FILE = REPO / "results" / "mc_channel_departure.json"

# First zero of J1: the radial relaxation mode decays as
# exp(-beta_1^2 t / tau_r), so tau_r/beta_1^2 is the relaxation eigentime.
BETA_1 = 3.8317059702


def _load():
    case = yaml.safe_load(CASE_FILE.read_text())
    metr = {m["name"]: m for m in case["metrics"]}
    return case, metr


def test_mc_channel_exact_kinematics():
    case, metr = _load()
    n = int(case["study"]["exact_particles"])
    v_mean = float(case["physical"]["mean_velocity"])
    c_x = float(case["receiver"]["axial_length"])

    reference_checks = ci.verify_limits()
    res = run_case(case, runner="langevin", n_particles=n, seed=0,
                   diffusivity=0.0)

    # The gate: the release sampler must draw P(r) = 2r/a^2.
    ks_floor = mc_error.ks_critical(n, alpha=metr["radial_ks"]["alpha"])
    assert res["ks_statistic"] < metr["radial_ks"]["tol_multiple_of_floor"] * ks_floor, (
        f"release sampler KS {res['ks_statistic']:.2e} above "
        f"{metr['radial_ks']['tol_multiple_of_floor']}x floor {ks_floor:.2e}"
    )

    receivers = []
    for rec in res["receivers"]:
        t, cm, co = rec["t"], rec["cir_measured"], rec["cir_reference"]

        # RMSE against the analytic reference sits in a BAND around the binomial floor:
        # the floor is exact in expectation, but one particle contributes at
        # every time, so the errors are correlated across times and the
        # measured RMSE scatters with O(1) relative spread. Too high is a
        # runner error; well below the floor would mean the comparison is
        # not independent of the analytic reference.
        rmse = METRICS["cir_rmse"](cm, co)
        floor = mc_error.binomial_rmse_floor(co, n)
        lo_band, hi_band = metr["cir_rmse"]["band_of_floor"]
        assert lo_band < rmse / floor < hi_band, (
            f"dbar={rec['dbar']*1e6:.0f}um: RMSE/floor = {rmse/floor:.2f} "
            f"outside [{lo_band}, {hi_band}]"
        )

        # Pre-onset counts are EXACTLY zero: no particle outruns the
        # centreline, so this is a kinematic impossibility, not a tolerance.
        pre = cm[t < rec["t1"]]
        assert pre.size > 0 and float(np.max(pre)) == 0.0, (
            f"dbar={rec['dbar']*1e6:.0f}um: {np.count_nonzero(pre)} pre-onset "
            f"samples nonzero — a particle arrived before t1 at D = 0"
        )

        # Peak within its binomial sigma.
        z_peak = (float(np.max(cm)) - rec["peak_value"]) / mc_error.binomial_sigma(
            rec["peak_value"], n)
        assert abs(z_peak) < metr["cir_peak_relative"]["tol_sigma"], (
            f"dbar={rec['dbar']*1e6:.0f}um: peak at {z_peak:+.2f} sigma"
        )

        # Tail mass over [t2, t_max] against the log law, within the
        # Minkowski upper bound on the integral's sd (conservative by
        # construction — see mc_error.binomial_integral_sigma_bound).
        sel = t >= rec["t2"]
        mass_m = float(np.trapezoid(cm[sel], t[sel]))
        mass_o = ci.tail_mass(rec["t2"], float(t[-1]), v_mean, rec["dbar"], c_x)
        bound = mc_error.binomial_integral_sigma_bound(t[sel], co[sel], n)
        assert abs(mass_m - mass_o) < metr["cir_tail_mass_relative"]["tol_sigma"] * bound, (
            f"dbar={rec['dbar']*1e6:.0f}um: tail mass off by "
            f"{(mass_m-mass_o)/bound:.2f}x its sigma bound"
        )

        receivers.append({
            "dbar_um": rec["dbar"] * 1e6,
            "t1_s": rec["t1"],
            "t2_s": rec["t2"],
            "peak_reference": rec["peak_value"],
            "peak_z_sigma": z_peak,
            "rmse_over_binomial_floor": rmse / floor,
            "tail_mass_measured": mass_m,
            "tail_mass_reference": mass_o,
            "tail_mass_sigma_bound": bound,
            "pre_onset_max": float(np.max(pre)),
        })

    record = {
        "case": case["name"],
        "leg": "exact kinematics (D = 0)",
        "reference_checks": reference_checks,
        "ks_statistic": res["ks_statistic"],
        "ks_floor": ks_floor,
        "receivers": receivers,
        "meta": res["meta"],
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")


def test_mc_channel_departure():
    case, metr = _load()
    study = case["study"]
    n = int(study["departure_particles"])

    res = run_case(case, runner="langevin", n_particles=n, seed=1,
                   time_horizon_over_t2=float(study["departure_horizon_over_t2"]))
    meta = res["meta"]
    tau_r = meta["tau_r"]

    # The gate now exercises the WALL SCHEME: reflection bias would distort
    # P(r) and bias the CIR invisibly through the velocity sampling.
    ks_floor = mc_error.ks_critical(n, alpha=metr["radial_ks"]["alpha"])
    assert res["ks_statistic"] < metr["radial_ks"]["tol_multiple_of_floor"] * ks_floor

    receivers = []
    for rec in res["receivers"]:
        t, cm, co = rec["t"], rec["cir_measured"], rec["cir_reference"]

        # Onset softening: axial diffusion delivers arrivals before t1.
        near_onset = cm[(t >= 0.90 * rec["t1"]) & (t < rec["t1"])]
        assert near_onset.size > 0 and float(np.max(near_onset)) > 0.0, (
            f"dbar={rec['dbar']*1e6:.0f}um: no pre-onset arrivals at D > 0"
        )

        # Peak depression: diffusive smoothing lowers the maximum.
        peak_ratio = float(np.max(cm)) / rec["peak_value"]
        assert peak_ratio < 1.0, (
            f"dbar={rec['dbar']*1e6:.0f}um: measured peak {peak_ratio:.3f} of "
            f"analytic reference — smoothing should depress it"
        )

        receivers.append({
            "dbar_um": rec["dbar"] * 1e6,
            "flow_dominated_ratio": rec["flow_dominated_ratio"],
            "peak_measured_over_reference": peak_ratio,
            "pre_onset_max": float(np.max(near_onset)),
        })

    # The two-regime tail, asserted on the MIDDLE receiver (dbar = 750 um),
    # whose crossover the standard horizon straddles.
    mid = next(r for r in res["receivers"] if abs(r["dbar"] - 750e-6) < 1e-9)
    t, cm, co, t2 = mid["t"], mid["cir_measured"], mid["cir_reference"], mid["t2"]

    # Regime 1 — ENHANCEMENT over [3, 5] t2: the upstream reservoir
    # (mass ~ dbar/c_x times the window population) feeds the window faster
    # than radial diffusion clears it. Significant, not just a sign: the
    # excess must exceed twice the Minkowski sigma bound.
    sel = (t >= 3.0 * t2) & (t <= 5.0 * t2)
    excess = float(np.trapezoid(cm[sel] - co[sel], t[sel]))
    bound = mc_error.binomial_integral_sigma_bound(t[sel], co[sel], n)
    assert excess > 2.0 * bound, (
        f"tail enhancement {excess:.2e} below 2x its sigma bound {bound:.2e} "
        f"— the reservoir-pumping regime should dominate here"
    )

    # Regime 2 — TERMINATION: by 12 t2 (~0.12 tau_r) every particle has
    # cleared the window, while the log-divergent analytic reference still predicts
    # ~1e-2. Measured zero at seed 1; the assertion allows a straggler.
    k12 = int(np.argmin(np.abs(t - 12.0 * t2)))
    assert cm[k12] < 0.2 * co[k12], (
        f"measured CIR at 12 t2 is {cm[k12]:.2e} against analytic reference {co[k12]:.2e} "
        f"— the tail should have terminated"
    )

    # The crossover, still recorded in eigentime units for continuity with
    # the earlier records. The eigentime ATTRIBUTION is withdrawn: the
    # pre-registered Pe sweep (results/eigentime_pe_sweep.json) measured
    # t_cross = K tau_r^0.31 dbar^0.73 — the layer-escape scaling — and
    # the 0.95-of-eigentime value below is a coincidence of this
    # parameter point.
    beyond = t > 2.0 * t2
    ratio = cm[beyond] / np.maximum(co[beyond], 1e-300)
    t_cross = float(t[beyond][np.argmax(ratio < 1.0)])

    record = {
        "case": case["name"],
        "leg": "departure (D > 0)",
        "receivers": receivers,
        "middle_receiver_tail": {
            "enhancement_excess_3_to_5_t2": excess,
            "enhancement_sigma_bound": bound,
            "cir_at_12_t2_measured": float(cm[k12]),
            "cir_at_12_t2_reference": float(co[k12]),
            "crossover_t_over_tau_r": t_cross / tau_r,
            "crossover_t_beta1sq_over_tau_r": t_cross * BETA_1**2 / tau_r,
            "crossover_clock": (
                "eigentime attribution WITHDRAWN by the pre-registered Pe "
                "sweep (results/eigentime_pe_sweep.json): measured "
                "t_cross = K tau_r^0.31 dbar^0.73, the layer-escape "
                "scaling; the 0.95-of-eigentime value here is a "
                "coincidence of this parameter point"
            ),
        },
        "prediction_correction": (
            "The pre-measurement prediction was a DEPLETED tail. Measured: "
            "enhancement first (reservoir pumping), termination after. "
            "Recorded in place per the correction policy."
        ),
        "meta": meta,
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    DEPARTURE_FILE.parent.mkdir(exist_ok=True)
    DEPARTURE_FILE.write_text(json.dumps(record, indent=2) + "\n")
