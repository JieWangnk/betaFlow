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
        "final_periodicity": meta["periodicity"][-1],
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
