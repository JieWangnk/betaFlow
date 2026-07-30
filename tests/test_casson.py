"""Casson two-axis study: mesh x regularisation cap (nuMax).

A yield-stress model is singular (nu -> inf as shear rate -> 0), so OpenFOAM
caps the viscosity at nuMax. The plug is therefore never rigid — it creeps —
and its computed width is set by where nu saturates rather than by the
physics. nuMax is a NON-PHYSICAL PARAMETER THAT CHANGES THE ANSWER, which is
why this case needs two refinement axes instead of one.

The conservation identity tau_w = G_disc * h follows from force balance ALONE
and is therefore rheology-independent: it is both the strongest single check
here and, because it compares two quantities that must agree at the fixed
point (rather than measuring iteration-to-iteration change), the honest
convergence gate. Residual and profile drift are change measures, and a
slowly-contracting nonlinear fixed point makes them look converged long
before the solution is.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.analytic import casson
from betaflow.metrics import METRICS
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "casson_steady.yaml"
RESULTS_FILE = REPO / "results" / "casson_steady.json"

MESH_LEVELS = (40, 80, 160)
GRID_ITERATIONS = 20000

# Extra mesh level at the ONLY cap ratio these meshes can resolve (N_min = 50),
# to show the plug-width bias reaching its continuum limit. Needs a bigger
# budget: convergence cost grows roughly as N^2 * nuMax_ratio.
FINE_LEVEL = 320
FINE_ITERATIONS = 120000
RESOLVABLE_RATIO = 1.0e2

# Identity is exact in the discrete balance; the band reflects linear-solver
# tolerances. Asserted on the sub-grid the cost model says must converge
# within GRID_ITERATIONS — if the transport model were mis-wired, it would
# fail here too, since the identity does not depend on the rheology.
IDENTITY_TOL = 1.0e-9
MUST_CONVERGE = ((40, 1.0e2), (40, 1.0e3), (80, 1.0e2), (160, 1.0e2))

# Measured plug creep is sampled at CELL CENTRES, so it misses the outermost
# sliver of the plug: the expected shortfall is exactly (y_last/y_p)^2 for the
# quadratic creep profile. Asserted against that, not against a fitted number.
#
# PRECONDITION: the creep model gammadot = G y / nuMax holds only where the
# cap is actually active. When the mesh cannot resolve the plug (N below
# n_min_measurable) the outermost cells used by the measurement lie OUTSIDE
# the cap-active region, where the true shear is far larger, and the measured
# creep is inflated (up to 1.9x here). Those points are reported but not
# asserted — the model is not wrong there, it simply does not apply.
CREEP_RTOL = 0.05


def _measure(case, n_cells, nu_max_ratio, iterations):
    h = float(case["geometry"]["half_height"])
    xi = float(case["nondim"]["xi"])
    result = run_case(
        case,
        runner="openfoam",
        n_cells=n_cells,
        sampling="cell",
        nu_max_ratio=nu_max_ratio,
        iterations=iterations,
        workdir=REPO / "_runs",
        require_converged=False,  # convergence is judged on the identity below
    )
    meta = result["meta"]
    g_disc = meta["pressure_gradient"]
    y = np.asarray(result["y"])
    u = np.asarray(result["u"])

    # y_p from the DISCRETE pressure gradient the solver actually applied.
    y_p = casson.plug_half_width(meta["tau0"], g_disc)
    identity = METRICS["wss_relative"](result["tau_w"], casson.tau_wall(g_disc, h))
    cap_width = METRICS["plug_width_cap_active"](y, result["nu_eff"], meta["nu_max"])
    flat_width = METRICS["plug_width_flatness"](y, u)
    creep = METRICS["plug_velocity_variation"](y, u, y_p)

    # Cell-centre quantisation of the creep measurement (see CREEP_RTOL).
    inside = np.abs(y)[np.abs(y) < y_p]
    y_last = inside.max() if inside.size else float("nan")
    quantisation = (y_last / y_p) ** 2 if inside.size else float("nan")

    return {
        "n_cells": n_cells,
        "nu_max_ratio": nu_max_ratio,
        "iterations": iterations,
        "identity": identity,
        "converged": bool(identity < 1.0e-6),
        "ux_residual": meta["ux_residual"],
        "measurement_drift": meta["measurement_drift"],
        "L2_velocity": METRICS["L2_velocity"](
            u / result["u_ref"], casson.velocity_profile(y / h, xi)
        ),
        "y_p": y_p,
        "plug_width_cap_active": cap_width,
        "plug_width_flatness": flat_width,
        "cap_bias": cap_width / y_p - 1.0,
        "cap_bias_predicted": casson.cap_active_half_width(
            meta["tau0"], g_disc, meta["nu_c"], meta["nu_max"]
        )
        / y_p
        - 1.0,
        "cap_bias_leading_order": casson.plug_width_relative_bias(
            meta["nu_c"], meta["nu_max"]
        ),
        "creep": creep,
        "creep_predicted": casson.plug_velocity_variation(
            meta["tau0"], g_disc, meta["nu_max"]
        )
        / result["u_ref"],
        "creep_cell_quantisation": quantisation,
        # Does the cap-active region cover every cell the creep measurement
        # uses? If not, the creep model's precondition fails (see CREEP_RTOL).
        "creep_cells_capped": bool(cap_width >= y_last),
        "cell_quantum_over_y_p": (2.0 * h / n_cells) / y_p,
        # Measurability floor: the cap defect eps*y_p must exceed a cell.
        "n_min_measurable": 1.0 / (xi * math.sqrt(meta["nu_c"] / meta["nu_max"])),
    }


@pytest.mark.slow
def test_casson_two_axis_study():
    case = yaml.safe_load(CASE_FILE.read_text())
    ratios = [float(r) for r in case["regularisation"]["nu_max_ratios"]]
    tols = {m["name"]: float(m["tol"]) for m in case["metrics"]}

    grid = [
        _measure(case, n, ratio, GRID_ITERATIONS)
        for n in MESH_LEVELS
        for ratio in ratios
    ]
    grid.append(_measure(case, FINE_LEVEL, RESOLVABLE_RATIO, FINE_ITERATIONS))

    def at(n, ratio):
        return next(
            g for g in grid if g["n_cells"] == n and g["nu_max_ratio"] == ratio
        )

    # Mesh convergence of the plug-width bias at the one resolvable ratio.
    bias_series = [
        {
            "n_cells": n,
            "cap_bias": at(n, RESOLVABLE_RATIO)["cap_bias"],
            "cell_quantum_over_y_p": at(n, RESOLVABLE_RATIO)["cell_quantum_over_y_p"],
        }
        for n in (*MESH_LEVELS, FINE_LEVEL)
    ]
    bias_limit = at(FINE_LEVEL, RESOLVABLE_RATIO)["cap_bias_predicted"]

    # Does the L2 error converge under mesh refinement, or floor on nuMax?
    order = {}
    for ratio in ratios:
        errs = [at(n, ratio)["L2_velocity"] for n in MESH_LEVELS]
        order[f"{ratio:.0e}"] = [
            math.log2(c / f) for c, f in zip(errs, errs[1:])
        ]

    record = {
        "case": case["name"],
        "runner": "openfoam",
        "xi": case["nondim"]["xi"],
        "mesh_levels": list(MESH_LEVELS),
        "nu_max_ratios": ratios,
        "grid_iterations": GRID_ITERATIONS,
        "grid": grid,
        "plug_bias_mesh_convergence": {
            "nu_max_ratio": RESOLVABLE_RATIO,
            "series": bias_series,
            "continuum_limit": bias_limit,
        },
        "L2_mesh_order_by_ratio": order,
        "identity_tol": IDENTITY_TOL,
        "converged_count": sum(g["converged"] for g in grid),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openfoam_version": "14",
        "git_sha": git_sha(REPO),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    # --- assertions (artifacts already on disk) -----------------------------

    # 1. The rheology-independent identity, on the sub-grid that must converge.
    for n, ratio in MUST_CONVERGE:
        entry = at(n, ratio)
        assert entry["identity"] < IDENTITY_TOL, (
            f"conservation identity tau_w = G_disc*h violated at N={n}, "
            f"nuMax/nu_c={ratio:.0e}: {entry['identity']:.3e} — this is "
            f"rheology-independent, so it indicates the transport model is "
            f"wired in wrongly, not that the Casson parameters are wrong"
        )

    # 2. Residual creep matches G y_p^2 / (2 nuMax), allowing for the
    #    cell-centre quantisation of the measurement — wherever the cap-active
    #    region actually covers the measured cells (see CREEP_RTOL).
    asserted_creep = [g for g in grid if g["converged"] and g["creep_cells_capped"]]
    assert asserted_creep, "no run satisfied the creep model's preconditions"
    for entry in asserted_creep:
        expected = entry["creep_predicted"] * entry["creep_cell_quantisation"]
        assert abs(entry["creep"] / expected - 1.0) < CREEP_RTOL, (
            f"plug creep at N={entry['n_cells']}, "
            f"nuMax/nu_c={entry['nu_max_ratio']:.0e} is {entry['creep']:.4e}, "
            f"off the quantisation-corrected prediction {expected:.4e} "
            f"by {abs(entry['creep'] / expected - 1):.1%}"
        )

    # 2b. The creep EXPONENT, measured: one decade of nuMax must divide the
    #     creep by ten. Both points have the cap covering the measured cells.
    #     (The higher-ratio point's global identity is ~1e-4 rather than
    #     round-off; creep is a local quantity and is converged there.)
    decade = at(160, ratios[0])["creep"] / at(160, ratios[1])["creep"]
    assert 9.5 < decade < 10.5, (
        f"plug creep should scale as 1/nuMax (factor 10 per decade); "
        f"measured factor {decade:.3f} between nuMax/nu_c="
        f"{ratios[0]:.0e} and {ratios[1]:.0e} at N=160"
    )

    # 3. The plug-width bias approaches its continuum limit from below as the
    #    mesh resolves it (monotone, and within one cell quantum at the end).
    biases = [b["cap_bias"] for b in bias_series]
    assert all(c < f for c, f in zip(biases, biases[1:])), (
        f"plug-width bias not monotone in mesh level: {biases}"
    )
    finest = bias_series[-1]
    assert bias_limit - finest["cap_bias"] < finest["cell_quantum_over_y_p"], (
        f"plug-width bias {finest['cap_bias']:.4f} at N={FINE_LEVEL} is further "
        f"than one cell quantum from the continuum limit {bias_limit:.4f}"
    )

    # 4. The regularisation floor: at the lowest cap ratio the L2 error stops
    #    converging under mesh refinement, while a higher cap keeps converging.
    assert order[f"{ratios[0]:.0e}"][-1] < 0.5, (
        f"expected the L2 error to floor on the regularisation at "
        f"nuMax/nu_c={ratios[0]:.0e}; observed order {order[f'{ratios[0]:.0e}'][-1]:.2f}"
    )
    assert order[f"{ratios[1]:.0e}"][-1] > 1.5, (
        f"expected the L2 error to still converge at nuMax/nu_c={ratios[1]:.0e}; "
        f"observed order {order[f'{ratios[1]:.0e}'][-1]:.2f}"
    )

    # 5. The case's own declared tolerances, on its best-resolved converged run.
    best = at(MESH_LEVELS[-1], RESOLVABLE_RATIO)
    assert best["L2_velocity"] < tols["L2_velocity"], (
        f"L2 velocity {best['L2_velocity']:.3e} over tol {tols['L2_velocity']:.0e}"
    )
    assert best["identity"] < tols["wss_relative"], (
        f"identity {best['identity']:.3e} over tol {tols['wss_relative']:.0e}"
    )
