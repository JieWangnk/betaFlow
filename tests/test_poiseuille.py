"""Validation test: steady plane Poiseuille vs the OpenFOAM runner.

Loads the YAML case, runs it through the solver-agnostic run_case() contract,
evaluates the configured metric against the analytic oracle, asserts the
tolerance, and logs a provenance-stamped record to results/.
"""

import json
import subprocess
from datetime import datetime, timezone
from importlib import import_module
from pathlib import Path

import numpy as np
import yaml

from betaflow.metrics import METRICS
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "poiseuille_steady.yaml"
RESULTS_FILE = REPO / "results" / "poiseuille_steady.json"

# Cells across the full channel height (2h). The L2 error is pure O(dy^2)
# discretisation + profile-interpolation error: 5.18e-3 / 1.30e-3 / 3.25e-4 at
# N = 20/40/80 (ratio 4.0 per doubling). N=80 meets the 1e-3 tolerance with 3x
# margin; N=40 does not.
MESH_LEVEL = 80


def _resolve(dotted):
    module, _, attr = dotted.rpartition(".")
    return getattr(import_module(module), attr)


def _git_sha():
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "--short=12", "HEAD"], cwd=REPO, text=True
        ).strip()
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD"], cwd=REPO).returncode != 0
        return sha + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def test_poiseuille_steady():
    case = yaml.safe_load(CASE_FILE.read_text())
    oracle = _resolve(case["oracle"])

    result = run_case(case, runner="openfoam", n_cells=MESH_LEVEL, workdir=REPO / "_runs")
    meta = result.get("meta", {})

    # Guard against the classic silent failure: the runner converting Re into a
    # viscosity with a different (velocity, length) convention than the oracle.
    h = float(case["geometry"]["half_height"])
    poiseuille = import_module("betaflow.analytic.poiseuille")
    np.testing.assert_allclose(
        poiseuille.reynolds(meta["u_mean"], h, meta["nu"]),
        case["nondim"]["Re"],
        rtol=1e-12,
        err_msg="runner and oracle disagree on the Reynolds-number definition",
    )

    y_over_h = np.asarray(result["y"]) / h
    u_nondim = np.asarray(result["u"]) / result["u_ref"]
    u_exact = oracle(y_over_h)

    spec = case["metrics"][0]
    error = METRICS[spec["name"]](u_nondim, u_exact)
    tol = float(spec["tol"])
    passed = bool(error < tol)

    record = {
        "case": case["name"],
        "runner": "openfoam",
        "mesh_level": MESH_LEVEL,
        "n_cells": meta.get("n_cells_total"),
        "error": error,
        "tol": tol,
        "passed": passed,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openfoam_version": meta.get("openfoam_version"),
        "git_sha": _git_sha(),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    assert passed, (
        f"L2 velocity error {error:.3e} exceeds tol {tol:.1e} "
        f"(mesh_level={MESH_LEVEL}, case dir: {meta.get('case_dir')})"
    )
