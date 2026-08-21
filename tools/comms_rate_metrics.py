#!/usr/bin/env python3
"""Communications metrics from the measured CIR: ISI, memory, rate.

Turns the channel model's curves into the numbers communications
engineering trades in, for the flow-dominated ANALYTIC MODEL and the
MEASURED channel (the Langevin departure run at physical diffusion,
re-run here deterministically — same seed and configuration as
results/mc_channel_departure.json) side by side.

Definitions (on-off keying, worst case = all previous bits are 1):
  signal        CIR at the detection instant t_d (the measured peak time)
  ISI(T_s)      sum over k >= 1 of CIR(t_d + k T_s): what earlier puffs
                still contribute at this slot's detection instant
  ISI ratio     ISI / signal, each side using its own curve
  memory        slots until a puff's contribution ends. MEASURED: finite,
                ceil(t_cross / T_s) with t_cross from the cumulative-
                excess extractor. MODEL: INFINITE — the 1/t tail summed
                over bits grows as (c_x / 2 V T_s) ln K, so the model
                cannot define a worst-case ISI at any rate; every model
                number below carries the SAME truncation as the data
                window, and says so.
  max rate      1 / T_s at the largest T_s-independent threshold crossing:
                the smallest T_s whose ISI ratio stays at or under the
                threshold (0.1 and 0.2 reported).

PRE-REGISTERED DIRECTION (from the two-act tail, before computing): for
short symbol intervals the interfering slots sample the ENHANCED part of
the measured tail, so measured ISI exceeds the (equally truncated) model;
for long intervals the slots land beyond the measured termination, where
the model's tail persists, so the model exceeds the measurement. The
crossover interval between the regimes is recorded per receiver.
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

THRESHOLDS = (0.1, 0.2)


def crossover_time(t, cm, co, t2):
    """Cumulative-excess extractor (the Pe sweep's settled method)."""
    valid = t > t2
    dm = (cm - co)[valid]
    tt = t[valid]
    cum = np.concatenate([[0.0], np.cumsum(np.diff(tt) * 0.5 * (dm[1:] + dm[:-1]))])
    return float(tt[int(np.argmax(cum))])


def isi_curve(t, curve, t_d, t_symbol, t_end):
    """Sum of interpolated CIR at t_d + k T_s, zero beyond the data window.

    Zero-extension is JUSTIFIED for the measured curve (it terminates
    within the window — that is the measured fact); for the model it is a
    stated truncation, because the untruncated model diverges.
    """
    total, k = 0.0, 1
    while t_d + k * t_symbol <= t_end:
        total += float(np.interp(t_d + k * t_symbol, t, curve))
        k += 1
    return total


def main():
    case = yaml.safe_load(
        (REPO / "betaflow" / "cases" / "mc_channel.yaml").read_text())
    v_mean = float(case["physical"]["mean_velocity"])
    c_x = float(case["receiver"]["axial_length"])
    res = run_case(case, runner="langevin", n_particles=30000, seed=1,
                   time_horizon_over_t2=float(
                       case["study"]["departure_horizon_over_t2"]))

    receivers = []
    for rec in res["receivers"]:
        t = np.asarray(rec["t"])
        cm = np.asarray(rec["cir_measured"])
        co = np.asarray(rec["cir_reference"])
        t2 = rec["t2"]
        t_end = float(t[-1])
        t_d = float(t[int(np.argmax(cm))])
        sig_m = float(np.max(cm))
        sig_o = float(ci.cir(t_d, v_mean, rec["dbar"], c_x))
        t_cross = crossover_time(t, cm, co, t2)

        sweep = []
        # Sweep to 12 t2 (the first pass stopped at 3 t2 and no threshold
        # crossed anywhere); capped so at least one model term fits the
        # data window.
        for ts_over_t2 in np.linspace(0.3, 12.0, 40):
            t_s = ts_over_t2 * t2
            if t_d + t_s > t_end:
                break
            isi_m = isi_curve(t, cm, t_d, t_s, t_end)
            isi_o = isi_curve(t, co, t_d, t_s, t_end)
            sweep.append({
                "t_symbol_over_t2": round(float(ts_over_t2), 4),
                "isi_ratio_measured": isi_m / sig_m,
                "isi_ratio_model_truncated": isi_o / sig_o,
                "memory_symbols_measured":
                    ci.channel_memory_symbols(t_cross, t_s),
                "model_divergence_coefficient_per_lnK":
                    ci.isi_divergence_coefficient(t_s, v_mean, c_x),
            })

        def max_rate(key, theta):
            xs = [row["t_symbol_over_t2"] for row in sweep]
            ys = [row[key] for row in sweep]
            for i in range(len(xs) - 1, 0, -1):
                if ys[i] <= theta < ys[i - 1]:
                    w = (theta - ys[i]) / (ys[i - 1] - ys[i])
                    ts = xs[i] - w * (xs[i] - xs[i - 1])
                    return 1.0 / (ts * t2)
            return None  # threshold never crossed inside the sweep

        # Where the measured and (truncated) model ISI curves cross —
        # the boundary between the under- and over-estimation regimes.
        diff = [row["isi_ratio_measured"] - row["isi_ratio_model_truncated"]
                for row in sweep]
        regime_cross = None
        for i in range(1, len(diff)):
            if diff[i - 1] > 0.0 >= diff[i]:
                regime_cross = sweep[i]["t_symbol_over_t2"]
                break

        receivers.append({
            "dbar_um": rec["dbar"] * 1e6,
            "t2_s": t2,
            "t_detect_s": t_d,
            "signal_measured": sig_m,
            "signal_model_at_t_detect": sig_o,
            "t_cross_s": t_cross,
            "sweep": sweep,
            "max_rate_hz": {
                str(th): {"measured": max_rate("isi_ratio_measured", th),
                          "model_truncated":
                              max_rate("isi_ratio_model_truncated", th)}
                for th in THRESHOLDS},
            "regime_crossover_t_symbol_over_t2": regime_cross,
        })

    record = {
        "definitions": "see module docstring; worst-case OOK, detection at "
                       "the measured peak; model numbers truncated at the "
                       "data window because the untruncated model diverges",
        "pre_registered_direction": (
            "short T_s: measured ISI above the equally-truncated model "
            "(enhanced tail); long T_s: below it (terminated tail); the "
            "regime crossover per receiver is recorded"),
        "direction_correction_measured": (
            "the pre-registered slot-position picture was TOO SIMPLE and is "
            "kept per the correction policy. The ISI sum spans the whole "
            "tail at once, so the BALANCE of enhancement (finite extra "
            "mass) against termination (all model mass beyond t_cross is "
            "phantom) decides the direction per receiver: the truncated "
            "model OVERSTATES ISI for the near and middle receivers at "
            "every sampled T_s, while the far receiver's enhancement "
            "outweighs its (late) termination at short T_s. Consequence "
            "at the thresholds: the model UNDERSTATES the achievable rate "
            "wherever its phantom tail dominates."),
        "receivers": receivers,
        "meta": {"n_particles": 30000, "seed": 1,
                 "source_run": "same configuration as "
                               "results/mc_channel_departure.json"},
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out = REPO / "results" / "comms_rate_metrics.json"
    out.write_text(json.dumps(record, indent=2) + "\n")

    for r in receivers:
        print(f"receiver {r['dbar_um']:.0f} um: t_cross {r['t_cross_s']:.3f} s, "
              f"regime crossover at T_s = {r['regime_crossover_t_symbol_over_t2']} t2")
        for th in THRESHOLDS:
            m = r["max_rate_hz"][str(th)]
            fm = f"{m['measured']:.2f}" if m["measured"] else "none in sweep"
            fo = (f"{m['model_truncated']:.2f}"
                  if m["model_truncated"] else "none in sweep")
            print(f"  max rate at ISI<= {th}: measured {fm} Hz, "
                  f"model(truncated) {fo} Hz")
    print(f"written: {out}")


if __name__ == "__main__":
    main()
