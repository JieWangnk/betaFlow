"""Null test: steady plane Couette at three mesh levels, both sampling modes.

The exact profile is linear, and every operator in the chain — the
second-order interior scheme, the half-cell one-sided wall gradient, and
linear (cellPoint) interpolation — is exact for linear fields. So unlike the
Poiseuille case there is no discretisation error to converge: every error at
EVERY mesh level must sit at round-off (floors: the 1e-9 steady-convergence
gate and 12-digit ASCII field I/O). A deviation is a bug in the framework or
solver, not a mesh effect, and is diagnosed rather than tolerated.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from betaflow.analytic import couette, resolve
from betaflow.metrics import METRICS
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "couette_steady.yaml"
RESULTS_FILE = REPO / "results" / "couette_steady.json"

MESH_LEVELS = (40, 80, 160)
SAMPLINGS = ("cellPoint", "cell")


def test_couette_steady_null():
    case = yaml.safe_load(CASE_FILE.read_text())
    reference = resolve(case["reference"])
    height = float(case["geometry"]["height"])
    tols = {m["name"]: float(m["tol"]) for m in case["metrics"]}

    openfoam_version = None
    velocity_errors = {s: [] for s in SAMPLINGS}
    tau_errors = []
    for sampling in SAMPLINGS:
        for level in MESH_LEVELS:
            result = run_case(
                case, runner="openfoam", n_cells=level, workdir=REPO / "_runs", sampling=sampling
            )
            meta = result["meta"]
            openfoam_version = meta["openfoam_version"]

            # Guard against a silent Re-definition mismatch, as in every case.
            np.testing.assert_allclose(
                couette.reynolds(meta["u_wall"], height, meta["nu"]),
                case["nondim"]["Re"],
                rtol=1e-12,
                err_msg="runner and analytic reference disagree on the Reynolds-number definition",
            )
            # Opt-out check: no force -> no pressure-gradient provenance.
            assert "pressure_gradient" not in meta, (
                "couette has no mean-force drive; pressure_gradient provenance should be absent"
            )

            y_over_height = np.asarray(result["y"]) / height
            u_nondim = np.asarray(result["u"]) / result["u_ref"]
            velocity_errors[sampling].append(
                METRICS["L2_velocity"](u_nondim, reference(y_over_height))
            )
            # tau_w comes from the solve, not the sampling; log it once.
            if sampling == "cell":
                tau_exact = couette.tau_wall(meta["nu"], meta["u_wall"], height)
                tau_errors.append(METRICS["wss_relative"](result["tau_w"], tau_exact))

    worst_velocity = max(max(errs) for errs in velocity_errors.values())
    worst_tau = max(tau_errors)
    passed = bool(
        worst_velocity < tols["L2_velocity"] and worst_tau < tols["wss_relative"]
    )

    record = {
        "case": case["name"],
        "runner": "openfoam",
        "mesh_levels": list(MESH_LEVELS),
        "null_test": True,
        "samplings": {s: {"L2_velocity": velocity_errors[s]} for s in SAMPLINGS},
        "tau_w": {"wss_relative": tau_errors},
        "tols": tols,
        "passed": passed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openfoam_version": openfoam_version,
        "git_sha": git_sha(REPO),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    assert passed, (
        f"null test not at round-off: worst L2_velocity {worst_velocity:.3e} "
        f"(tol {tols['L2_velocity']:.0e}), worst wss_relative {worst_tau:.3e} "
        f"(tol {tols['wss_relative']:.0e}) — a deviation here is a bug, "
        f"not discretisation error"
    )
