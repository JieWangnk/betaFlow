"""The off-axis Ma^2 tensor: 50-digit algebra against an actual lattice.

The analytic reference derives D_ab = (tau - 1/2)(Pi^eq_ab/C - u_a u_b) for
any flow direction (lattice_boltzmann docstring section 7, verified against
the exact amplification matrix to 5.6e-17). This test is the measurement
leg: a 2-D Gaussian blob advected obliquely on D2Q5, the growth of its
second-moment tensor read as 2 D_ab.

What each run pins, beyond "the numbers match":

  o1_oblique   The cross component D_xy = -(tau - 1/2) u_x u_y is NEGATIVE
               and reproduced by the lattice — oblique advection generates
               anticorrelated spreading, invisible to every 1-D experiment
               this repo ran before.
  o2_oblique   The second-order equilibrium — which cancels the axis-aligned
               depletion at these weights — does NOT remove the cross term:
               D2Q5 has no diagonal velocity to carry M4_xxyy, so the fix
               cannot reach it. This is the anisotropy budget line for any
               future geometry whose flow crosses the lattice obliquely
               (the bifurcation's daughter branches).
  o1_axis      The depletion is rank-one along the flow: D_xx depleted by
               u^2 (tau - 1/2), D_yy exactly undepleted. The naive
               isotropic-depletion reading fails here by construction.

The record keeps predictions and measurements side by side; predictions
come from the analytic module at run time, never re-derived here.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

from betaflow.analytic import lattice_boltzmann as lb
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "lbm_offaxis.yaml"
RESULTS_FILE = REPO / "results" / "lbm_offaxis_tensor.json"


def test_lbm_offaxis_tensor():
    case = yaml.safe_load(CASE_FILE.read_text())
    tols = {m["name"]: float(m["tol"]) for m in case["metrics"]}
    tol = tols["tensor_component_relative"]

    runs = []
    for spec in case["study"]["runs"]:
        r = run_case(case, runner="lbm",
                     tau=float(spec["tau"]), u=tuple(spec["u"]),
                     equilibrium_order=int(spec["equilibrium_order"]))
        m, p = r["d_measured"], r["d_tensor_predicted"]
        rel = {k: abs(m[k] - p[k]) / abs(p["xx"]) for k in ("xx", "yy", "xy")}
        for k, v in rel.items():
            assert v < tol, (
                f"{spec['tag']}: D_{k} measured {m[k]:+.8f} vs tensor "
                f"prediction {p[k]:+.8f}, relative {v:.2e} > {tol:.0e}"
            )
        assert r["mass_drift"] < tols["mass_drift"], (
            f"{spec['tag']}: mass drift {r['mass_drift']:.2e}"
        )
        runs.append({
            "tag": spec["tag"],
            "tau": float(spec["tau"]),
            "u": [float(v) for v in spec["u"]],
            "equilibrium_order": int(spec["equilibrium_order"]),
            "d_measured": m,
            "d_tensor_predicted": p,
            "d_isotropic_naive": r["d_isotropic_naive"],
            "worst_component_relative": max(rel.values()),
            "mass_drift": r["mass_drift"],
        })

    by_tag = {r["tag"]: r for r in runs}

    # The named alternatives, excluded by measurement rather than assumed:
    # (a) isotropic depletion — the axis run's D_yy must be UNdepleted ...
    ax = by_tag["o1_axis"]
    d_iso = ax["d_isotropic_naive"]
    assert abs(ax["d_measured"]["yy"] - d_iso) / d_iso < tol, (
        "transverse diffusivity is depleted; the tensor law says it must "
        f"not be: D_yy {ax['d_measured']['yy']:.8f} vs c_s^2(tau-1/2) {d_iso:.8f}"
    )
    depletion = 0.20**2 * (0.8 - 0.5)
    assert abs((d_iso - ax["d_measured"]["xx"]) - depletion) / d_iso < tol, (
        "flow-parallel depletion is not u^2 (tau - 1/2)"
    )
    # ... (b) "the second-order fix removes the error" — the cross term must
    # survive order 2 at full size.
    o2 = by_tag["o2_oblique"]
    cross_expected = -(1.0 - 0.5) * 0.15 * 0.10
    assert abs(o2["d_measured"]["xy"] - cross_expected) / abs(cross_expected) < 1e-4, (
        "the cross depletion should survive the second-order equilibrium "
        f"on D2Q5: measured {o2['d_measured']['xy']:.2e} vs {cross_expected:.2e}"
    )

    record = {
        "case": case["name"],
        "runner": "lbm",
        "law": "D_ab = (tau - 1/2)(Pi_eq_ab/C - u_a u_b), "
               "lattice_boltzmann.diffusion_tensor (docstring section 7)",
        "runs": runs,
        "named_alternatives_excluded": {
            "isotropic_depletion": "o1_axis measures D_yy undepleted at the "
                                   "gate while D_xx is depleted by exactly "
                                   "u^2(tau-1/2)",
            "second_order_fix_suffices": "o2_oblique measures the cross "
                                         "component at full size with the "
                                         "second-order equilibrium applied",
        },
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")
