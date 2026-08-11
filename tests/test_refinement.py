"""Code verification: observed order of accuracy for poiseuille_steady.

With an exact solution in hand this is an order-of-accuracy test, not GCI:
the true discretisation error is measured at three mesh levels and the
observed order p = log2(e_coarse / e_fine) is compared against the formal
order of the scheme (2). GCI-style solution verification is reserved for
cases without an analytic reference.

Runs both sampling modes — cellPoint-interpolated profile extraction and raw
cell values — so interpolation error is separated from solver error. Logs
results/poiseuille_steady_refinement.json and generates
report/order_of_accuracy.md.
"""

import json
import math
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
RESULTS_FILE = REPO / "results" / "poiseuille_steady_refinement.json"
REPORT_FILE = REPO / "report" / "order_of_accuracy.md"

MESH_LEVELS = (40, 80, 160)

SAMPLINGS = {
    "cellPoint": "lineUniform stations, cellPoint interpolation (default profile extraction)",
    "cell": "lineCell, raw cell values (discrete solution, no interpolation)",
}

# The formal order of the discretisation is 2; accept the standard +/-0.2 band
# on the observed order. Declared a priori — never widened to make a run pass.
P_BAND = (1.8, 2.2)

FORMAL_ORDER = 2

# --- Wall shear stress: the mechanism is locked, not just the rate. ---
# tau_w = G_disc*h identically (discrete momentum balance pins the wall flux),
# so its entire error is G_disc vs G_exact. For the discrete-mean bulk
# constraint on a uniform mesh this is EXACT, not leading-order:
#     G_disc/G_exact = 1/(1 + 2/N^2)  =>  rel error = (2/N^2)/(1 + 2/N^2)
# If meanVelocityForce's constraint definition ever changes, p survives but
# this coefficient moves — the model band is the guard that catches it.
TAU_P_BAND = (1.9, 2.1)
TAU_MODEL_RTOL = 0.05


def _tau_error_model(n):
    return (2.0 / n**2) / (1.0 + 2.0 / n**2)


def test_order_of_accuracy():
    case = yaml.safe_load(CASE_FILE.read_text())
    reference = resolve(case["reference"])
    h = float(case["geometry"]["half_height"])
    metric_name = case["metrics"][0]["name"]
    metric = METRICS[metric_name]

    openfoam_version = None
    study = {}
    tau_errors = []
    identity_devs = []
    for sampling in SAMPLINGS:
        errors, n_cells = [], []
        for level in MESH_LEVELS:
            result = run_case(
                case, runner="openfoam", n_cells=level, workdir=REPO / "_runs", sampling=sampling
            )
            y_over_h = np.asarray(result["y"]) / h
            u_nondim = np.asarray(result["u"]) / result["u_ref"]
            errors.append(metric(u_nondim, reference(y_over_h)))
            n_cells.append(result["meta"]["n_cells_total"])
            openfoam_version = result["meta"]["openfoam_version"]
            # Conservation identity: the discrete momentum balance pins the
            # wall flux to the applied mean pressure-gradient source exactly.
            tau_identity = result["meta"]["pressure_gradient"] * h
            identity_devs.append(abs(result["tau_w"] - tau_identity) / tau_identity)
            # tau_w comes from the solve, not the profile sampling, so one
            # sampling mode's runs suffice to measure its convergence.
            if sampling == "cell":
                meta = result["meta"]
                tau_exact = poiseuille.tau_wall(
                    poiseuille.pressure_gradient(meta["u_mean"], h, meta["nu"]), h
                )
                tau_errors.append(METRICS["wss_relative"](result["tau_w"], tau_exact))
        p = [math.log2(coarse / fine) for coarse, fine in zip(errors, errors[1:])]
        study[sampling] = {"n_cells": n_cells, "errors": errors, "p": p}

    tau_p = [math.log2(coarse / fine) for coarse, fine in zip(tau_errors, tau_errors[1:])]
    tau_predicted = [_tau_error_model(n) for n in MESH_LEVELS]

    record = {
        "case": case["name"],
        "runner": "openfoam",
        "metric": metric_name,
        "mesh_levels": list(MESH_LEVELS),
        "formal_order": FORMAL_ORDER,
        "samplings": study,
        "tau_w": {
            "metric": "wss_relative",
            "errors": tau_errors,
            "p": tau_p,
            "predicted_errors": tau_predicted,
            "error_model": "(2/N^2)/(1+2/N^2) — exact for the discrete-mean bulk constraint",
            "conservation_max_rel_dev": max(identity_devs),
        },
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openfoam_version": openfoam_version,
        "git_sha": git_sha(REPO),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")
    _write_report(record)

    # Artifacts are written above whatever happens next: a failing order test
    # must still leave the evidence on disk.
    for sampling, data in study.items():
        errs = data["errors"]
        assert all(c > f for c, f in zip(errs, errs[1:])), (
            f"[{sampling}] error does not decrease monotonically: {errs}"
        )
        for pair, p in zip(zip(MESH_LEVELS, MESH_LEVELS[1:]), data["p"]):
            assert P_BAND[0] < p < P_BAND[1], (
                f"[{sampling}] observed order p={p:.3f} outside {P_BAND} "
                f"for levels {pair[0]}->{pair[1]} (formal order {FORMAL_ORDER})"
            )

    # tau_w: three locks, ordered weakest to sharpest.
    # (1) Observed order in the accepted band.
    for pair, p in zip(zip(MESH_LEVELS, MESH_LEVELS[1:]), tau_p):
        assert TAU_P_BAND[0] < p < TAU_P_BAND[1], (
            f"tau_w observed order p={p:.3f} outside {TAU_P_BAND} for levels {pair[0]}->{pair[1]}"
        )
    # (2) The coefficient, not just the rate: error must track the exact
    # constraint model. p would survive a changed meanVelocityForce
    # definition; this would not.
    for level, err, pred in zip(MESH_LEVELS, tau_errors, tau_predicted):
        assert abs(err / pred - 1.0) < TAU_MODEL_RTOL, (
            f"tau_w error {err:.4e} at N={level} deviates "
            f"{abs(err / pred - 1):.1%} from the model {pred:.4e} (band {TAU_MODEL_RTOL:.0%})"
        )
    # (3) The conservation identity itself, tau_w = G_disc*h. Exact in the
    # discrete balance; the tolerance reflects the 1e-9 steady-convergence
    # gate and the ASCII round-trip, not the identity.
    assert max(identity_devs) < 1e-9, (
        f"conservation identity tau_w = G_disc*h violated: "
        f"max relative deviation {max(identity_devs):.3e}"
    )


def _write_report(record):
    lines = [
        "# Observed order of accuracy — poiseuille_steady",
        "",
        "Code verification of the OpenFOAM runner against the exact plane-Poiseuille",
        "solution: the true discretisation error is measured at three mesh levels and",
        "the observed order p = log2(e(N) / e(2N)) is compared to the formal order of",
        f"the scheme ({record['formal_order']}). Refinement ratio 2 throughout.",
        "",
        f"Generated by tests/test_refinement.py — do not edit by hand.",
        "",
        f"- case: `{record['case']}`, runner: `{record['runner']}` "
        f"(OpenFOAM {record['openfoam_version']})",
        f"- metric: `{record['metric']}` (normalised by u_max)",
        f"- timestamp: {record['timestamp']}",
        f"- git: `{record['git_sha']}`",
        "",
    ]
    for sampling, description in SAMPLINGS.items():
        data = record["samplings"][sampling]
        lines += [
            f"## {description}",
            "",
            "| mesh level | n_cells | L2 error | p |",
            "|---:|---:|---:|---:|",
        ]
        for i, level in enumerate(record["mesh_levels"]):
            p = f"{data['p'][i - 1]:.3f}" if i > 0 else "—"
            lines.append(f"| {level} | {data['n_cells'][i]} | {data['errors'][i]:.3e} | {p} |")
        lines.append("")
    tau = record["tau_w"]
    lines += [
        "## Wall shear stress (relative error vs exact tau_w = G h)",
        "",
        "| mesh level | rel error | model (2/N²)/(1+2/N²) | p |",
        "|---:|---:|---:|---:|",
    ]
    for i, level in enumerate(record["mesh_levels"]):
        p = f"{tau['p'][i - 1]:.3f}" if i > 0 else "—"
        lines.append(
            f"| {level} | {tau['errors'][i]:.4e} | {tau['predicted_errors'][i]:.4e} | {p} |"
        )
    lines += [
        "",
        "tau_w = G_disc·h identically (discrete momentum balance; conservation "
        f"identity holds to {tau['conservation_max_rel_dev']:.1e} across all runs), "
        "so its error is entirely G_disc vs G — second order despite the formally "
        "O(dy) one-sided wall snGrad.",
        "",
    ]
    REPORT_FILE.parent.mkdir(exist_ok=True)
    REPORT_FILE.write_text("\n".join(lines))
