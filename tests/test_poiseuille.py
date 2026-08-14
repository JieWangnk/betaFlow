"""Validation test: steady plane Poiseuille vs the OpenFOAM runner.

Loads the YAML case, runs it through the solver-independent run_case() contract,
evaluates the configured metric against the analytic reference, asserts the
tolerance, and logs a provenance-stamped record to results/.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from betaflow.analytic import poiseuille, resolve
from betaflow.metrics import METRICS
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "poiseuille_steady.yaml"
RESULTS_FILE = REPO / "results" / "poiseuille_steady.json"

# Cells across the full channel height (2h). The L2 error is pure O(dy^2)
# discretisation + profile-interpolation error: 5.18e-3 / 1.30e-3 / 3.25e-4 at
# N = 20/40/80 (ratio 4.0 per doubling). N=80 meets the 1e-3 tolerance with 3x
# margin; N=40 does not. See results/poiseuille_steady_refinement.json.
MESH_LEVEL = 80


def test_poiseuille_steady():
    case = yaml.safe_load(CASE_FILE.read_text())
    reference = resolve(case["reference"])

    result = run_case(case, runner="openfoam", n_cells=MESH_LEVEL, workdir=REPO / "_runs")
    meta = result.get("meta", {})

    # Guard against the classic silent failure: the runner converting Re into a
    # viscosity with a different (velocity, length) convention than the analytic reference.
    h = float(case["geometry"]["half_height"])
    np.testing.assert_allclose(
        poiseuille.reynolds(meta["u_mean"], h, meta["nu"]),
        case["nondim"]["Re"],
        rtol=1e-12,
        err_msg="runner and analytic reference disagree on the Reynolds-number definition",
    )

    y_over_h = np.asarray(result["y"]) / h
    u_nondim = np.asarray(result["u"]) / result["u_ref"]
    u_exact = reference(y_over_h)

    # Exact wall shear stress in kinematic units, matching the runner's tau_w.
    tau_exact = poiseuille.tau_wall(poiseuille.pressure_gradient(meta["u_mean"], h, meta["nu"]), h)

    evaluations = {
        "L2_velocity": lambda: METRICS["L2_velocity"](u_nondim, u_exact),
        "wss_relative": lambda: METRICS["wss_relative"](result["tau_w"], tau_exact),
    }

    metric_records = []
    for spec in case["metrics"]:
        error = evaluations[spec["name"]]()
        tol = float(spec["tol"])
        metric_records.append(
            {"name": spec["name"], "error": error, "tol": tol, "passed": bool(error < tol)}
        )
    passed = all(m["passed"] for m in metric_records)

    record = {
        "case": case["name"],
        "runner": "openfoam",
        "mesh_level": MESH_LEVEL,
        "n_cells": meta.get("n_cells_total"),
        "metrics": metric_records,
        "passed": passed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openfoam_version": meta.get("openfoam_version"),
        "git_sha": git_sha(REPO),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    failures = [m for m in metric_records if not m["passed"]]
    assert passed, (
        f"metrics over tolerance: "
        + ", ".join(f"{m['name']}={m['error']:.3e} (tol {m['tol']:.1e})" for m in failures)
        + f" (mesh_level={MESH_LEVEL}, case dir: {meta.get('case_dir')})"
    )
