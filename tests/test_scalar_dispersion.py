"""Eulerian scalar dispersion, through the moment hierarchy.

FOUR ANCHORS, and the point of the case is that they fail differently.

  d_eff       the headline number, and the one a solver can hit by accident:
              numerical axial diffusion inflating D and a shear-sampling
              deficit deflating the U^2 term cancel in the sum.
  intercept   weights the SAME transverse spectrum as beta^-8 rather than
              beta^-6, so it is not absorbable into a fitted D_eff.
  kappa_3     exactly blind to axial diffusion, so it needs nothing
              subtracted; and its sign differs between the geometries.
  centroid    exactly U t at all times for a uniform release, independent of
              D, Pe, mesh and numerical diffusion.

The convergence assertion is made on the INTERCEPT, not on D_eff. D_eff is
already at 1e-9 relative at the coarsest level here and has no headroom left
to measure an order with — asserting a rate on it would be measuring
round-off. This is the same trap the couette null test documents.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.analytic import advection_diffusion as ad
from betaflow.metrics import METRICS
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "scalar_dispersion.yaml"
RESULTS_FILE = REPO / "results" / "scalar_dispersion.json"

P_BAND = (1.7, 2.3)


def _errors(case, result, window):
    return {
        "d_eff_relative": METRICS["d_eff_relative"](
            result["var_x"], result["t"], window, result["d_eff_expected"]
        ),
        "variance_intercept_relative": METRICS["variance_intercept_relative"](
            result["var_x"], result["t"], window, result["intercept_expected"]
        ),
        "cumulant_slope_relative": METRICS["cumulant_slope_relative"](
            result["third_cumulant"], result["t"], window,
            result["kappa3_slope_expected"],
        ),
        # Against the DISCRETE mean velocity: see the metric's docstring.
        "centroid_relative": METRICS["centroid_relative"](
            result["centroid"], result["t"],
            result["u_mean_discrete"],
            result["centroid_offset_expected"],
        ),
    }


def test_scalar_dispersion():
    case = yaml.safe_load(CASE_FILE.read_text())
    num, study = case["numerics"], case["study"]
    tols = {m["name"]: float(m["tol"]) for m in case["metrics"]}

    reference_checks = ad.verify_limits()
    record = {
        "case": case["name"],
        "runner": "moments",
        "note": "NO axial discretisation exists in this runner; it isolates "
                "the transverse operator and says nothing about axial advection",
        "reference_self_verification_checks": len(reference_checks),
        "reference_worst_check": max(reference_checks.values()),
        "geometries": {},
    }

    for geom in study["geometries"]:
        case_g = {**case, "geometry": {**case["geometry"], "type": geom}}
        tau = ad.transverse_relaxation_time(
            float(case["geometry"]["length_scale"]),
            float(case["physical"]["diffusivity"]),
        )
        window = [w * tau for w in num["fit_window_tau"]]

        runs = []
        for n in study["mesh_levels"]:
            r = run_case(case_g, runner="moments", n_cells=int(n),
                         n_steps=int(num["n_steps"]),
                         t_end_over_tau=float(num["t_end_over_tau"]))
            runs.append({
                "n_cells": int(n),
                "mass_drift": r["meta"]["mass_drift"],
                "operator_row_sum": r["meta"]["operator_row_sum"],
                **_errors(case_g, r, window),
            })

        # Point release, on the streamline where u = U exactly. The centroid
        # still ends up permanently BEHIND; "seeded at the mean velocity, so
        # no offset" is the plausible wrong answer this checks.
        xi0 = 1.0 / math.sqrt(2.0) if geom == "pipe" else 1.0 / math.sqrt(3.0)
        r_pt = run_case(case_g, runner="moments",
                        n_cells=int(study["mesh_levels"][-1]),
                        n_steps=int(num["n_steps"]),
                        t_end_over_tau=float(num["t_end_over_tau"]),
                        release="point", xi_release=xi0)

        icept = [x["variance_intercept_relative"] for x in runs]
        record["geometries"][geom] = {
            "runs": runs,
            "intercept_order": [
                round(math.log2(c / f), 3) for c, f in zip(icept, icept[1:])
            ],
            "dispersion_factor": ad.DISPERSION_FACTOR[geom],
            "skewness_factor": ad.SKEWNESS_FACTOR[geom],
            "balance_peclet": ad.balance_peclet(geom),
            "onset_symmetric": ad.asymptotic_onset(geom),
            "onset_asymmetric": ad.asymptotic_onset(geom, symmetric_release=False),
            "point_release": {
                "xi_requested": xi0,
                "xi_actual": r_pt["meta"]["xi_actual"],
                "velocity_deviation_there": float(
                    ad.velocity_deviation(xi0, geom)
                ),
                "offset_expected": r_pt["centroid_offset_expected"],
                "u_mean_discretisation_error": r_pt["meta"][
                    "u_mean_discretisation_error"
                ],
                "offset_measured": float(
                    r_pt["centroid"][-1]
                    - r_pt["u_mean_discrete"] * r_pt["t"][-1]
                ),
            },
        }

    record["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["git_sha"] = git_sha(REPO)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    # --- assertions --------------------------------------------------------
    for geom, block in record["geometries"].items():
        finest = block["runs"][-1]

        # 1. CONSERVATION, which is a property of the operator rather than of
        #    convergence: the row sums vanish identically, so m0 cannot drift.
        assert finest["operator_row_sum"] < 1e-12, (
            f"{geom}: transverse operator is not conservative "
            f"(row sum {finest['operator_row_sum']:.3e})"
        )
        for x in block["runs"]:
            assert x["mass_drift"] < 1e-9, (
                f"{geom}: m0 drifted by {x['mass_drift']:.3e} at N={x['n_cells']}"
            )

        # 2. The four anchors at the finest level.
        for name, tol in tols.items():
            assert finest[name] < tol, (
                f"{geom}: {name} = {finest[name]:.3e} exceeds {tol:.0e} "
                f"at N={finest['n_cells']}"
            )

        # 3. Second order on the INTERCEPT. Asserted here and not on d_eff,
        #    which is already at round-off and would measure nothing.
        for p in block["intercept_order"]:
            assert P_BAND[0] < p < P_BAND[1], (
                f"{geom}: intercept converges at p = {p}, outside {P_BAND}"
            )

        # 4. The point release sits exactly on the u = U streamline and still
        #    ends up behind. Both facts, or the anchor proves nothing.
        pr = block["point_release"]
        assert abs(pr["velocity_deviation_there"]) < 1e-12
        assert pr["offset_expected"] < 0.0, "the exact offset must be a LAG"
        assert pr["offset_measured"] < 0.0, "the measured offset must be a LAG"
        # Compared at the cell centre actually used, so this is physics
        # rather than a one-cell placement artefact.
        assert abs(pr["offset_measured"] / pr["offset_expected"] - 1.0) < 1e-3

    # 5. The two geometries must disagree where they should, or the geometry
    #    argument is being ignored somewhere in the chain.
    pipe, chan = record["geometries"]["pipe"], record["geometries"]["channel"]
    assert pipe["dispersion_factor"] != chan["dispersion_factor"]
    assert pipe["skewness_factor"] > 0.0 > chan["skewness_factor"], (
        "the third cumulant must change SIGN between the geometries"
    )


def test_scalar_dispersion_axial_scope_is_declared():
    """The runner must not claim to test what it cannot.

    It carries no axial mesh, so it cannot see an axial advection scheme. If
    that ever stops being true this test should be deleted deliberately, not
    silently outgrown.
    """
    case = yaml.safe_load(CASE_FILE.read_text())
    r = run_case(case, runner="moments", n_cells=40, n_steps=8)
    assert "NO axial discretisation" in r["meta"]["note"]
    # The transverse Peclet is the only Peclet this runner knows about.
    assert r["meta"]["peclet"] == pytest.approx(
        float(case["physical"]["mean_velocity"])
        * float(case["geometry"]["length_scale"])
        / float(case["physical"]["diffusivity"])
    )
