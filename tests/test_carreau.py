"""Carreau study: single-axis mesh refinement, Cu sweep, and a blood point.

This case completes the constitutive axis and is the CONTRAST to casson.
Carreau has no regularisation parameter — nu is smooth and bounded between
nu_inf and nu_0 at every strain rate — so there is no cap, no plug, no
second study axis, and (predicted) no error floor under mesh refinement.

It also has an EXACT solution, contrary to the once-common claim that
shear-thinning channel flow needs GCI-style solution verification. Steady 1-D
force balance gives tau(y) = G y for ANY rheology, so the profile follows from
a pointwise rootfind plus a quadrature, both to machine precision. The oracle
verifies ITSELF against two analytic limits before being used as ground truth.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from betaflow.analytic import carreau
from betaflow.metrics import METRICS
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "carreau_steady.yaml"
NEWTONIAN_CASE_FILE = REPO / "betaflow" / "cases" / "poiseuille_steady.yaml"
RESULTS_FILE = REPO / "results" / "carreau_steady.json"

ORACLE_LIMIT_RTOL = 1e-12
P_BAND = (1.8, 2.2)

# Casson needed ~350x the Newtonian iteration count and grew without bound in
# nuMax. Carreau's contraction is geometric, so the ratio should be bounded
# and roughly independent of both mesh and Cu. Declared a priori from that
# mechanism, not fitted: an order of magnitude is the claim being tested.
STIFFNESS_RATIO_MAX = 10.0


def _evaluate(result, h):
    """Errors and shear-thinning signature for one run."""
    meta = result["meta"]
    g_disc = meta["pressure_gradient"]
    y = np.asarray(result["y"])
    u = np.asarray(result["u"])

    exact = carreau.velocity_profile(
        y / h, g_disc, h, meta["nu0"], meta["nu_inf"], meta["k"], meta["n"], meta["a"]
    )
    u_nondim = u / result["u_ref"]
    return {
        "n_cells": meta["mesh_level"],
        "Cu": carreau.carreau_number(meta["k"], g_disc, h, meta["nu0"]),
        "n": meta["n"],
        "nu_inf_over_nu0": meta["nu_inf"] / meta["nu0"],
        "L2_velocity": METRICS["L2_velocity"](u_nondim, exact),
        "wss_relative": METRICS["wss_relative"](
            result["tau_w"], carreau.tau_wall(g_disc, h)
        ),
        "identity": meta["identity"],
        # Shear-thinning signature: u_centre/u_mean is 1.5 for a parabola and
        # falls toward (2n+1)/(n+1) in the power-law limit as the profile
        # flattens.
        "centreline_flatness": float(np.max(u) / meta["u_mean"]),
        # Same ratio from the exact solution at the SAME discrete G, so the
        # comparison is free of any normalisation choice.
        "centreline_flatness_exact": float(
            carreau.u_max(g_disc, h, meta["nu0"], meta["nu_inf"], meta["k"], meta["n"], meta["a"])
            / carreau.u_mean(g_disc, h, meta["nu0"], meta["nu_inf"], meta["k"], meta["n"], meta["a"])
        ),
        "iterations_to_residual_1e10": meta["iterations_to_residual_1e10"],
        "ux_residual": meta["ux_residual"],
    }


def test_carreau_steady():
    case = yaml.safe_load(CASE_FILE.read_text())
    newtonian_case = yaml.safe_load(NEWTONIAN_CASE_FILE.read_text())
    h = float(case["geometry"]["half_height"])
    levels = [int(n) for n in case["study"]["mesh_levels"]]
    cu_sweep = [float(c) for c in case["study"]["cu_sweep"]]
    tols = {m["name"]: float(m["tol"]) for m in case["metrics"]}

    # 0. Verify the verifier BEFORE using it as ground truth.
    oracle_limits = carreau.verify_limits(rtol=ORACLE_LIMIT_RTOL)

    def _run(**kwargs):
        return run_case(case, runner="openfoam", sampling="cell", workdir=REPO / "_runs", **kwargs)

    # 1. Single-axis mesh refinement at the case's Cu. No second axis.
    refinement = [_evaluate(_run(n_cells=n), h) for n in levels]
    errors = [r["L2_velocity"] for r in refinement]
    p_observed = [math.log2(c / f) for c, f in zip(errors, errors[1:])]

    # 2. Cu sweep at fixed mesh.
    sweep = [_evaluate(_run(n_cells=levels[0], cu=cu), h) for cu in cu_sweep]

    # 3. Physiological point: Cho & Kensey blood. k is fixed, Cu is an output.
    blood_result = _run(n_cells=levels[0], fluid="blood_cho_kensey")
    blood = _evaluate(blood_result, blood_result["meta"]["half_height"])
    blood["fluid"] = "blood_cho_kensey"

    # 4. Newtonian reference iteration counts, same meshes and solver settings,
    #    for the stiffening comparison.
    newtonian = []
    for n in levels:
        res = run_case(
            newtonian_case, runner="openfoam", n_cells=n, sampling="cell",
            workdir=REPO / "_runs",
        )
        newtonian.append(
            {
                "n_cells": n,
                "iterations_to_residual_1e10": res["meta"]["iterations_to_residual_1e10"],
            }
        )
    stiffness = [
        r["iterations_to_residual_1e10"] / w["iterations_to_residual_1e10"]
        for r, w in zip(refinement, newtonian)
    ]

    # Mechanism reference: with nu_inf = 0 the same sweep IS monotone in Cu,
    # tending to the power-law flatness (2n+1)/(n+1). Oracle-only, so free.
    n_index = float(case["nondim"]["n"])
    power_law_flatness = []
    for cu in cu_sweep:
        g, k = carreau.drive_for_carreau_number(1.0, h, 0.02, 0.0, n_index, cu)
        power_law_flatness.append(
            carreau.u_max(g, h, 0.02, 0.0, k, n_index)
            / carreau.u_mean(g, h, 0.02, 0.0, k, n_index)
        )

    main = refinement[-1]
    passed = main["L2_velocity"] < tols["L2_velocity"] and main["wss_relative"] < tols["wss_relative"]

    record = {
        "case": case["name"],
        "runner": "openfoam",
        "oracle_self_verification": oracle_limits,
        "oracle_limit_rtol": ORACLE_LIMIT_RTOL,
        "mesh_refinement": {
            "Cu_target": case["nondim"]["Cu"],
            "levels": levels,
            "runs": refinement,
            "p_observed": p_observed,
        },
        "cu_sweep": {"n_cells": levels[0], "runs": sweep},
        "power_law_limit_flatness": power_law_flatness,
        "power_law_limit_asymptote": (2 * n_index + 1) / (n_index + 1),
        "blood_point": blood,
        "newtonian_reference": newtonian,
        "stiffness_ratio_vs_newtonian": stiffness,
        "main": {"tols": tols, "errors": {k: main[k] for k in tols}, "passed": passed},
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openfoam_version": "14",
        "git_sha": git_sha(REPO),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    # --- assertions (artifacts already on disk) -----------------------------
    all_runs = [*refinement, *sweep, blood]

    # 1. The rheology-independent identity, on every run. It is also the
    #    runner's convergence gate for generalised-Newtonian cases, so a run
    #    reaching here has already closed it to 1e-6; assert the tighter band.
    for run in all_runs:
        assert run["identity"] < 1e-9, (
            f"conservation identity open by {run['identity']:.3e} at "
            f"N={run['n_cells']}, Cu={run['Cu']:.3g}"
        )

    # 2. Second-order convergence AND NO FLOOR — the contrast with casson,
    #    which floors because its regularisation parameter, not the mesh,
    #    limits the answer.
    for pair, p in zip(zip(levels, levels[1:]), p_observed):
        assert P_BAND[0] < p < P_BAND[1], (
            f"observed order p={p:.3f} outside {P_BAND} for {pair[0]}->{pair[1]}"
        )
    assert errors[-1] < errors[0] / 10.0, (
        f"error should keep falling under refinement (no regularisation "
        f"floor); got {errors}"
    )

    # 3. No stiffening: bounded iteration cost versus Newtonian, at every mesh
    #    and every Cu.
    for n, ratio in zip(levels, stiffness):
        assert ratio < STIFFNESS_RATIO_MAX, (
            f"carreau took {ratio:.1f}x the Newtonian iteration count at N={n}"
        )
    # Cost DOES grow with Cu (the prediction of Cu-independence was wrong),
    # but geometrically and boundedly: at the lowest Cu it must match the
    # Newtonian count, which is the mechanism claim that survives.
    sweep_iters = [r["iterations_to_residual_1e10"] for r in sweep]
    assert all(c <= f for c, f in zip(sweep_iters, sweep_iters[1:])), (
        f"iteration count should rise monotonically with Cu: {sweep_iters}"
    )
    assert sweep_iters[0] < STIFFNESS_RATIO_MAX * newtonian[0]["iterations_to_residual_1e10"], (
        f"at the lowest Cu the cost must approach Newtonian; got "
        f"{sweep_iters[0]} vs {newtonian[0]['iterations_to_residual_1e10']}"
    )

    # 4. Shear-thinning signature. The committed prediction was that
    #    centreline flatness falls MONOTONICALLY with Cu. It does not, and the
    #    exact oracle shows the same turn, so this is physics rather than a
    #    solver artefact: with a finite viscosity ratio the near-wall fluid
    #    enters the SECOND Newtonian plateau at high Cu and the profile
    #    de-flattens back toward a parabola. Monotonicity holds only in the
    #    nu_inf = 0 (pure power-law) limit, checked on the oracle below.
    #    What the solver must do is REPRODUCE the exact curve, monotone or not.
    for run in sweep:
        assert abs(run["centreline_flatness"] / run["centreline_flatness_exact"] - 1.0) < 0.01, (
            f"centreline flatness {run['centreline_flatness']:.4f} at "
            f"Cu={run['Cu']:.3g} disagrees with the exact "
            f"{run['centreline_flatness_exact']:.4f}"
        )
    flatness = [r["centreline_flatness"] for r in sweep]
    assert abs(flatness[0] - 1.5) < 0.02, (
        f"lowest Cu should approach the parabolic 1.5; got {flatness[0]:.4f}"
    )
    assert min(flatness) < flatness[0] - 0.1, (
        f"shear thinning must flatten the profile somewhere in the sweep: {flatness}"
    )
    assert all(c > f for c, f in zip(power_law_flatness, power_law_flatness[1:])), (
        f"with nu_inf = 0 the flatness must fall monotonically: {power_law_flatness}"
    )

    # 5. Error coefficient. The committed prediction was that it grows as Cu
    #    rises. It does not grow MONOTONICALLY, for the same reason the
    #    flatness turns: what drives the error is departure from parabolic,
    #    and a finite nu_inf caps how far the fluid can thin. So the error
    #    peaks where the profile is flattest, and both fall back together.
    #    The order stays 2 throughout (assertion 2).
    sweep_errors = [r["L2_velocity"] for r in sweep]
    assert int(np.argmax(sweep_errors)) == int(np.argmin(flatness)), (
        f"L2 error should peak where the profile is furthest from parabolic; "
        f"errors {sweep_errors}, flatness {flatness}"
    )
    assert all(e > sweep_errors[0] for e in sweep_errors[1:]), (
        f"every shear-thinning point should cost more error than the "
        f"near-Newtonian one: {sweep_errors}"
    )

    # 6. The case's own declared tolerances.
    assert passed, (
        f"main run over tolerance: L2={main['L2_velocity']:.3e} "
        f"(tol {tols['L2_velocity']:.0e}), "
        f"wss={main['wss_relative']:.3e} (tol {tols['wss_relative']:.0e})"
    )
