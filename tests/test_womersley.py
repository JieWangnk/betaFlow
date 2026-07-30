"""Pulsatile plane-channel (Womersley) study: amplitude, phase, WSS, identity.

Four experiments in one provenance-stamped record:

1. alpha-sweep {5, 10, 20} at FIXED cells-per-Stokes-layer (mesh refines with
   alpha) — isolates the predicted high-alpha WSS amplification from
   under-resolution, which a fixed mesh conflates.
2. The fixed-mesh point (alpha=20, N=80) the committed prediction (~6e-3 WSS
   relative error) was stated for.
3. Order study at alpha=5 under combined space-time refinement (both the
   interior scheme and `backward` are formally 2nd order).
4. One Euler run — if the time scheme is 1st order it dominates and masks
   the spatial order.

Per-timestep conservation identity tau_w(t) = h (G(t) - d<u>/dt) is asserted
for every run (discrete derivative matching the solver's ddt scheme).
Periodicity (cycle-to-cycle, tol 1e-6, cap 10 cycles) is REPORTED, not
asserted: the homogeneous-mode decay timescale (2/pi^3) alpha^2 T makes the
criterion unreachable within 10 cycles at high alpha from any accessible
initial state — the recorded per-cycle ratios document exactly where each
run lands. First measurement: observed orders are reported, not banded.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from betaflow.analytic import womersley
from betaflow.metrics import METRICS
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "womersley_pulsatile.yaml"
RESULTS_FILE = REPO / "results" / "womersley_pulsatile.json"

SWEEP_ALPHAS = (5.0, 10.0, 20.0)
ORDER_LEVELS = ((32, 64), (64, 128), (128, 256))  # (n_cells, n_steps) pairs
ORDER_ALPHA = 5.0

IDENTITY_TOL = 1e-6  # bounded by linear-solver tolerances, not the identity

# Phase-metric calibration: Euler at fixed mesh, refining only in time. The
# semi-discrete theory replaces i*omega by the scheme's symbol lambda, so the
# phase error is predictable in closed form — if the metric measures the time
# scheme and nothing else, measurement matches prediction.
CALIBRATION_STEPS = (64, 128, 256)
CALIBRATION_CELLS = 64
# A-priori band: the symbol prediction neglects SPATIAL error, whose
# standalone phase contribution is measured by the backward arm of the same
# sweep (~4.5e-4, which plateaus once temporal error falls below it). That is
# under 4% of the Euler phase error at the finest level, so 5% is the honest
# theory-derived band — not a fitted one.
CALIBRATION_RTOL = 0.05


def _symbol_phase_error(n_steps, alpha, scheme, h=1.0, period=1.0):
    """Phase error predicted by the ddt scheme's discrete symbol.

    Semi-discrete in time: the scheme replaces i*omega by
    lambda = (1 - z)/dt (Euler) or (3 - 4z + z^2)/(2 dt) (backward),
    z = exp(-i omega dt), so the profile becomes
    (g/lambda)(1 - cosh(sqrt(lambda/nu) y)/cosh(sqrt(lambda/nu) h)).
    Both the 1/lambda prefactor (the leading omega*dt/2 lag for Euler) and
    the modified Stokes wavenumber contribute.
    """
    omega = 2.0 * np.pi / period
    nu = omega * h**2 / alpha**2
    dt = period / n_steps
    z = np.exp(-1.0j * omega * dt)
    lam = (1.0 - z) / dt if scheme == "Euler" else (3.0 - 4.0 * z + z * z) / (2.0 * dt)
    y = np.linspace(-0.999 * h, 0.999 * h, 400)
    k = np.sqrt(lam / nu)
    discrete = (1.0 / lam) * (1.0 - np.cosh(k * y) / np.cosh(k * h))
    exact = womersley.complex_profile(y / h, alpha)
    return METRICS["L2_phase"](
        np.angle(discrete), np.angle(exact), np.abs(exact) ** 2
    )


def _evaluate(result, alpha, h):
    """Amplitude/phase/WSS errors of one run against the oracle."""
    exact = womersley.complex_profile(np.asarray(result["y"]) / h, alpha)
    tau_exact = womersley.complex_wall_shear(alpha)
    amp_err = METRICS["L2_amplitude"](result["u_amp"] / result["u_ref"], np.abs(exact))
    phase_err = METRICS["L2_phase"](
        result["u_phase"], np.angle(exact), np.abs(exact) ** 2
    )
    wss_amp_err = METRICS["wss_amp_relative"](
        result["tau_amp"] / result["tau_ref"], np.abs(tau_exact)
    )
    wss_phase_err = float(
        np.abs(np.angle(np.exp(1j * (result["tau_phase"] - np.angle(tau_exact)))))
    )
    meta = result["meta"]
    # Per-cycle transient decay: once the fast Stokes-layer modes die (one
    # cycle), the ratios decay geometrically at the slowest channel mode's
    # rate exp(-nu pi^2 T / 4h^2) = exp(-pi^3 / 2 alpha^2). Logged in full —
    # the ratios ARE the evidence for the convergence claim, so storing only
    # the last value (as an earlier revision did) discards it.
    ratios = np.asarray(meta["periodicity"])
    late = ratios[1:][-4:]
    decay_observed = float(np.mean(late[1:] / late[:-1])) if late.size > 1 else None
    return {
        "alpha": alpha,
        "n_cells": meta["mesh_level"],
        "cells_per_stokes_layer": round(meta["cells_per_stokes_layer"], 3),
        "n_steps": meta["n_steps_per_cycle"],
        "ddt": meta["ddt"],
        "L2_amplitude": amp_err,
        "L2_phase": phase_err,
        "wss_amp_relative": wss_amp_err,
        "wss_phase_error": wss_phase_err,
        "cycles_to_periodic": meta["cycles_to_periodic"],
        "periodicity": meta["periodicity"],
        "decay_observed": decay_observed,
        "decay_predicted": float(np.exp(-np.pi**3 / (2.0 * alpha**2))),
        "identity_max_rel": meta["identity_max_rel"],
    }


def test_womersley_pulsatile():
    case = yaml.safe_load(CASE_FILE.read_text())
    h = float(case["geometry"]["half_height"])
    tols = {m["name"]: float(m["tol"]) for m in case["metrics"]}

    openfoam_version = None
    identity_worst = 0.0

    def _run(**kwargs):
        nonlocal openfoam_version, identity_worst
        result = run_case(case, runner="openfoam", workdir=REPO / "_runs", **kwargs)
        openfoam_version = result["meta"]["openfoam_version"]
        identity_worst = max(identity_worst, result["meta"]["identity_max_rel"])
        return result

    # 1. alpha-sweep at fixed cells per Stokes layer (case-default resolution).
    sweep = []
    for alpha in SWEEP_ALPHAS:
        sweep.append(_evaluate(_run(alpha=alpha), alpha, h))

    # 2. The committed prediction's own normalisation: fixed mesh N=80.
    fixed_mesh = _evaluate(_run(alpha=20.0, n_cells=80), 20.0, h)

    # 3. Combined space-time order study (backward, formally 2nd order both).
    order = [
        _evaluate(_run(alpha=ORDER_ALPHA, n_cells=n, n_steps=nt), ORDER_ALPHA, h)
        for n, nt in ORDER_LEVELS
    ]
    p_amplitude = [
        math.log2(c["L2_amplitude"] / f["L2_amplitude"]) for c, f in zip(order, order[1:])
    ]
    p_wss = [
        math.log2(c["wss_amp_relative"] / f["wss_amp_relative"])
        for c, f in zip(order, order[1:])
    ]

    # 4. Euler at the middle level: 1st-order time should dominate everything.
    euler = _evaluate(
        _run(alpha=ORDER_ALPHA, n_cells=ORDER_LEVELS[1][0], n_steps=ORDER_LEVELS[1][1], ddt="Euler"),
        ORDER_ALPHA,
        h,
    )

    # 5. Phase-metric calibration: fixed mesh, time refinement only.
    calibration = []
    for scheme in ("Euler", "backward"):
        for nt in CALIBRATION_STEPS:
            run = _evaluate(
                _run(alpha=ORDER_ALPHA, n_cells=CALIBRATION_CELLS, n_steps=nt, ddt=scheme),
                ORDER_ALPHA,
                h,
            )
            predicted = _symbol_phase_error(nt, ORDER_ALPHA, scheme)
            calibration.append(
                {
                    "ddt": scheme,
                    "n_steps": nt,
                    "measured": run["L2_phase"],
                    "leading_order_pi_over_nt": math.pi / nt if scheme == "Euler" else None,
                    "symbol_prediction": predicted,
                    "ratio": run["L2_phase"] / predicted,
                }
            )

    main = sweep[-1]  # alpha = 20 at case-default resolution
    passed = all(main[name] < tol for name, tol in tols.items())

    record = {
        "case": case["name"],
        "runner": "openfoam",
        "main": {"tols": tols, "errors": {k: main[k] for k in tols}, "passed": passed},
        "sweep_fixed_cells_per_stokes_layer": sweep,
        "fixed_mesh_alpha20_N80": fixed_mesh,
        "order_study": {
            "alpha": ORDER_ALPHA,
            "levels": [list(l) for l in ORDER_LEVELS],
            "runs": order,
            "p_amplitude": p_amplitude,
            "p_wss": p_wss,
            "euler": euler,
        },
        "phase_metric_calibration": {
            "n_cells": CALIBRATION_CELLS,
            "alpha": ORDER_ALPHA,
            "rtol": CALIBRATION_RTOL,
            "runs": calibration,
        },
        "identity_tol": IDENTITY_TOL,
        "identity_worst": identity_worst,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openfoam_version": openfoam_version,
        "git_sha": git_sha(REPO),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    # Artifacts are on disk; now the assertions.
    assert identity_worst < IDENTITY_TOL, (
        f"per-timestep momentum-balance identity violated: worst {identity_worst:.3e}"
    )
    failures = [n for n, tol in tols.items() if main[n] >= tol]
    assert passed, (
        "main case (alpha=20, fixed cells/delta) over tolerance: "
        + ", ".join(f"{n}={main[n]:.3e} (tol {tols[n]:.0e})" for n in failures)
    )
    for key in ("L2_amplitude", "wss_amp_relative"):
        errs = [r[key] for r in order]
        assert all(c > f for c, f in zip(errs, errs[1:])), (
            f"order study {key} not monotone: {errs}"
        )
    assert euler["L2_amplitude"] > order[1]["L2_amplitude"], (
        "Euler should be less accurate than backward at the same resolution"
    )
    # The phase metric must measure the time scheme's symbol and nothing else.
    # Asserted for Euler only: backward's temporal phase error falls below the
    # fixed mesh's spatial floor within this sweep (visible as a plateau in the
    # logged backward arm), so its ratio is not a metric-calibration signal.
    for entry in calibration:
        if entry["ddt"] != "Euler":
            continue
        assert abs(entry["ratio"] - 1.0) < CALIBRATION_RTOL, (
            f"phase metric off symbol prediction by {abs(entry['ratio'] - 1):.1%} "
            f"at n_steps={entry['n_steps']} (band {CALIBRATION_RTOL:.0%}): "
            f"measured {entry['measured']:.4e} vs predicted {entry['symbol_prediction']:.4e}"
        )
