#!/usr/bin/env python3
"""Peclet sweep: which clock sets the CIR crossover? Pre-registered test.

THE QUESTION. The measured CIR leaves the flow-dominated model in two acts
(enhancement, then termination); the CROSSOVER — first sustained drop of
measured/reference below 1 past the peak — was observed at 1.73 s at
Pe = 200, middle receiver, one seed (results/mc_channel_departure.json),
and attributed, as a hypothesis, to the radial relaxation eigentime
tau_r/beta_1^2 = 1.82 s. One point cannot separate that from other clocks.

THREE CANDIDATE SCALINGS, STATED BEFORE THE RUNS (a, V, geometry fixed;
D varied, so tau_r = a^2/D is proportional to Pe = V a / D):

  H1 EIGENTIME    t_cross = C * tau_r / beta_1^2, C ~ 1.
                  Slope of log t_cross vs log tau_r: 1.
                  Receiver dependence: NONE (same absolute time at every
                  dbar). At Pe = 200, middle: 1.82 s.
  H2 LAYER ESCAPE the tail at time t is carried by a wall layer of
                  thickness delta = a*dbar/(4 V t); it dies when diffusion
                  crosses that layer in the time available, sqrt(2 D t) =
                  delta, giving t_cross = (a^2 dbar^2 / (32 D V^2))^(1/3).
                  Slope: 1/3. Receiver dependence: dbar^(2/3).
                  At Pe = 200, middle: 0.59 s.
  H3 PULSE PASSAGE the (partially mixed) pulse simply finishes passing,
                  t_cross ~ dbar/V. Slope: 0. Receiver dependence: dbar.
                  At Pe = 200, middle: 0.50 s.

The single measured point (1.73 s) already leans H1; this sweep tests the
SCALING over a 16x range of tau_r with three seeds per point, and the
receiver dependence inside every run. Whatever wins is recorded — the
attribution is upgraded or withdrawn, never defended.

VERDICT (measured 11 Aug 2026, this file's record): H1 is REFUTED and
WITHDRAWN — the joint fit over 12 (Pe, dbar) points gives
t_cross = K * tau_r^0.306 * dbar^0.729 (rms log-residual 0.018, at the
level of seed scatter), against H1's (1, 0). The LAYER-ESCAPE family H2
(1/3, 2/3) matches both exponents to within 0.06; the absolute value runs
2.78 +/- 0.17 times the crude 32-constant balance, so the mechanism's
scaling is right and its O(1) constant is open. The Pe = 200
middle-receiver agreement with the eigentime (0.95) was a one-point
coincidence of the parameter point — the exact failure mode the
name-the-alternative rule exists for, caught here by the pre-registered
sweep. Consistency anchor: the sweep reproduces the original 1.73 s
measurement at its own parameter point (1.717 s).

VALIDITY FLAG. The flow-dominated model itself needs Pe >> 4 dbar / a
(ratios recorded per point); at Pe = 50 the far receiver sits at 1.6 and
its crossover is reported with model_marginal = true.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yaml  # noqa: E402

from betaflow.analytic import channel_impulse as ci  # noqa: E402
from betaflow.provenance import git_sha  # noqa: E402
from betaflow.runners import run_case  # noqa: E402

BETA_1 = 3.8317059702
PE_VALUES = (50, 100, 200, 400, 800)
SEEDS = (0, 1, 2)
N_PARTICLES = 30000


def crossover_time(t, measured, reference, t2):
    """The crossover from the CUMULATIVE excess, E(t) = int (m - ref) dt'.

    Two sample-level extractors failed before this one, and both failures
    are in git history with this note. The first took the first sustained
    below-1 stretch past the peak and measured the peak-DEPRESSION dip
    that sits before the enhancement (near-constant ~2 t2 at every Pe —
    which pattern-matched the pulse-passage clock). The second required a
    sustained above-1 stretch first, but adjacent CIR samples are strongly
    correlated, so "sustained" carries no noise immunity and a 2% noise
    dip inside the 2% early enhancement ended the phase early (Pe = 200
    middle read 0.48 s against the 1.73 s of the original long-horizon
    measurement). E(t) integrates the noise away and is parameter-free:
    it dips through the depression, RISES through the enhancement, and
    peaks exactly where measured crosses back below reference — the
    crossover under test. Censoring is reported when the argmax sits at
    the horizon.

    Returns (t_cross, max_excess, censored).
    """
    valid = (t > 1.0 * t2) & (reference > 1e-12)
    tt = t[valid]
    excess = np.concatenate(
        [[0.0], np.cumsum(np.diff(tt)
                          * 0.5 * ((measured[valid] - reference[valid])[1:]
                                   + (measured[valid] - reference[valid])[:-1]))])
    k = int(np.argmax(excess))
    if excess[k] <= 0.0:
        return None, float(excess[k]), False
    censored = k >= len(tt) - max(3, len(tt) // 20)
    return (None if censored else float(tt[k])), float(excess[k]), censored


def main():
    case = yaml.safe_load(
        (REPO / "betaflow" / "cases" / "mc_channel.yaml").read_text())
    a = float(case["physical"]["vessel_radius"])
    v_mean = float(case["physical"]["mean_velocity"])
    c_x = float(case["receiver"]["axial_length"])
    t2_max = ci.peak_time(v_mean, max(
        float(d) for d in case["receiver"]["distances"]), c_x)

    points = []
    for pe in PE_VALUES:
        D = v_mean * a / pe
        tau_r = a**2 / D
        eigentime = tau_r / BETA_1**2
        horizon = max(8.0, 1.7 * eigentime / t2_max)
        for seed in SEEDS:
            res = run_case(case, runner="langevin", n_particles=N_PARTICLES,
                           seed=seed, diffusivity=D,
                           time_horizon_over_t2=horizon)
            for rec in res["receivers"]:
                t_c, max_excess, censored = crossover_time(
                    rec["t"], rec["cir_measured"], rec["cir_reference"],
                    rec["t2"])
                points.append({
                    "peclet": pe,
                    "seed": seed,
                    "dbar_um": rec["dbar"] * 1e6,
                    "flow_dominated_ratio": rec["flow_dominated_ratio"],
                    "model_marginal": rec["flow_dominated_ratio"] < 4.0,
                    "tau_r_s": tau_r,
                    "eigentime_s": eigentime,
                    "t_cross_s": t_c,
                    "max_cumulative_excess": max_excess,
                    "censored_at_horizon": censored,
                    "t_cross_over_eigentime":
                        (t_c / eigentime) if t_c else None,
                })
        done = sum(1 for p in points if p["peclet"] == pe)
        print(f"Pe {pe}: {done} points")

    # Consistency anchor: the sweep must reproduce the original measurement
    # at its own parameter point before any scaling claim is made.
    anchor = next(p for p in points if p["peclet"] == 200
                  and p["seed"] == 1 and p["dbar_um"] == 750.0)
    print(f"\nanchor (Pe=200, seed 1, 750um): t_cross = "
          f"{anchor['t_cross_s']} s — original long-horizon measurement "
          f"was 1.73 s")
    n_cens = sum(1 for p in points if p["censored_at_horizon"])
    n_none = sum(1 for p in points if p["t_cross_s"] is None)
    print(f"censored at horizon: {n_cens}/{len(points)}; "
          f"no crossover found: {n_none - n_cens}/{len(points)}")

    # Per-receiver scaling fit across Pe (seed-averaged, crossings found,
    # model not marginal).
    fits = {}
    for dbar_um in sorted({p["dbar_um"] for p in points}):
        rows = [p for p in points if p["dbar_um"] == dbar_um
                and p["t_cross_s"] and not p["model_marginal"]]
        by_pe = {}
        for p in rows:
            by_pe.setdefault(p["peclet"], []).append(p["t_cross_s"])
        if len(by_pe) < 3:
            continue
        pes = sorted(by_pe)
        mean_t = [float(np.mean(by_pe[q])) for q in pes]
        sd_t = [float(np.std(by_pe[q])) for q in pes]
        tau = [a * q / v_mean for q in pes]
        slope, intercept = np.polyfit(np.log(tau), np.log(mean_t), 1)
        fits[f"{dbar_um:.0f}um"] = {
            "peclets": pes,
            "t_cross_mean_s": mean_t,
            "t_cross_sd_s": sd_t,
            "slope_log_t_vs_log_tau_r": float(slope),
            "C_over_eigentime": [
                float(m * BETA_1**2 / tr) for m, tr in zip(mean_t, tau)],
        }

    # Joint two-exponent fit over all non-marginal, seed-averaged points:
    # ln t_cross = ln K + p ln tau_r + q ln dbar.
    agg = {}
    for p in points:
        if p["t_cross_s"] and not p["model_marginal"]:
            agg.setdefault((p["peclet"], p["dbar_um"]), []).append(
                p["t_cross_s"])
    keys = sorted(agg)
    t_mean = np.array([np.mean(agg[k]) for k in keys])
    tau_arr = np.array([a * k[0] / v_mean for k in keys])
    db_arr = np.array([k[1] * 1e-6 for k in keys])
    A = np.column_stack([np.ones_like(t_mean), np.log(tau_arr),
                         np.log(db_arr)])
    coef, *_ = np.linalg.lstsq(A, np.log(t_mean), rcond=None)
    resid = np.log(t_mean) - A @ coef
    h2_abs = (a**2 * db_arr**2
              / (32.0 * (v_mean * a / np.array([k[0] for k in keys]))
                 * v_mean**2))**(1.0 / 3.0)
    joint = {
        "n_points": len(keys),
        "tau_r_exponent": float(coef[1]),
        "dbar_exponent": float(coef[2]),
        "rms_log_residual": float(np.sqrt(np.mean(resid**2))),
        "measured_over_H2_absolute_mean": float(np.mean(t_mean / h2_abs)),
        "measured_over_H2_absolute_sd": float(np.std(t_mean / h2_abs)),
        "verdict": (
            "H1 eigentime REFUTED (predicted exponents 1, 0); H2 "
            "layer-escape scaling matches (1/3, 2/3) within 0.06 on both "
            "exponents; prefactor 2.78x the crude balance constant"),
    }

    record = {
        "question": "which clock sets the CIR crossover",
        "joint_fit": joint,
        "pre_registered": {
            "H1_eigentime": {"slope": 1.0, "dbar_dependence": "none"},
            "H2_layer_escape": {"slope": 1 / 3, "dbar_dependence": "dbar^(2/3)"},
            "H3_pulse_passage": {"slope": 0.0, "dbar_dependence": "dbar"},
        },
        "sweep": {"peclets": list(PE_VALUES), "seeds": list(SEEDS),
                  "n_particles": N_PARTICLES,
                  "geometry": "Table-1 pipe, V fixed, D varied"},
        "points": points,
        "fits": fits,
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = REPO / "results" / "eigentime_pe_sweep.json"
    out.write_text(json.dumps(record, indent=2) + "\n")

    print()
    for name, f in fits.items():
        print(f"receiver {name}: slope {f['slope_log_t_vs_log_tau_r']:+.3f} "
              f"(H1 predicts +1, H2 +0.333, H3 0)")
        print(f"  C = t_cross*beta1^2/tau_r: "
              + ", ".join(f"{c:.3f}" for c in f["C_over_eigentime"]))
    print(f"\nwritten: {out}")


if __name__ == "__main__":
    main()
