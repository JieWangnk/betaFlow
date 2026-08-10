"""OpenFOAM 14 particle runner vs the exact Brownian/Taylor-Aris oracles.

The three-way check this closes: the oracles are exact, runners/langevin.py is
the independently validated reference (D_eff/D = 9.9997 vs 10.0 exact), and
this file runs the SAME cases through OpenFOAM 14 via the brownianTracer cloud
(runners/openfoam_particles.py) — the displacement-level walk written after
OF14's stock BrownianMotionForce measured D/D_SE = 0.38-0.59 depending on
maxCo. A disagreement here is attributable, because a trusted reference sits
between truth and solver.

Scope (agreed): free diffusion, and the Taylor-Aris LONG-TIME anchor with the
radial KS gate — the short-time t^2 study and the 9-point radius sweep stay
langevin-only.

Results go to *_openfoam.json files so the committed langevin records stay
byte-stable. Both tests are @slow: they need OpenFOAM 14 plus the built
libbrownianTracerCloud.so, and the pipe case runs ~3 minutes.

Gate order in the pipe test mirrors tests/test_taylor_aris.py: KS before
D_eff, because a wrong wall treatment biases D_eff invisibly. During
development the KS gate caught two wall defects in sequence (escape walls
absorbing particles; plain rebound un-reflected by recalculation -> sticky
walls, KS 3.31x floor, D_eff +8.3%).
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.metrics import METRICS, mc_error
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]

FREE_CASE = REPO / "betaflow" / "cases" / "langevin_free.yaml"
FREE_RESULTS = REPO / "results" / "langevin_free_openfoam.json"

TA_CASE = REPO / "betaflow" / "cases" / "taylor_aris.yaml"
TA_RESULTS = REPO / "results" / "taylor_aris_openfoam.json"

RUNNER = "openfoam_particles"


@pytest.mark.slow
def test_langevin_free_openfoam():
    case = yaml.safe_load(FREE_CASE.read_text())
    n = int(case["study"]["reference_particles"])

    r = run_case(case, runner=RUNNER, n_particles=n, seed=0,
                 workdir=REPO / "_runs")
    err = METRICS["msd_slope_relative"](r["msd"], r["t"], r["D_expected"])

    sigma_coeff = float(case["metrics"][0]["sigma_coefficient"])
    tol_sigma = float(case["metrics"][0]["tol_sigma"])
    sigma = sigma_coeff / math.sqrt(n)

    cx, cy, cz = r["msd_components"][-1]
    isotropy = (max(cx, cy, cz) - min(cx, cy, cz)) / np.mean([cx, cy, cz])

    record = {
        "case": case["name"],
        "runner": RUNNER,
        "solver": r["meta"]["solver"],
        "note": "OpenFOAM 14 modular Lagrangian, brownianTracer cloud - "
                "displacement-level walk; stock BrownianMotionForce measured "
                "D/D_SE = 0.38-0.59 (maxCo-dependent) on this machine",
        "openfoam_version": r["meta"]["openfoam_version"],
        "cloud_library": r["meta"]["cloud_library"],
        "diffusivity": r["meta"]["diffusivity"],
        "stokes_number": r["meta"]["stokes_number"],
        "rms_displacement_m": r["meta"]["rms_displacement"],
        "reference": {
            "n_particles": n,
            "slope_error": float(err),
            "sigma_theory": sigma,
            "tol_sigma": tol_sigma,
            "tol_absolute": tol_sigma * sigma,
            "z_score": float(err) / sigma,
        },
        "isotropy_spread": float(isotropy),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(REPO),
    }
    FREE_RESULTS.write_text(json.dumps(record, indent=2) + "\n")

    assert err < tol_sigma * sigma, (
        f"OpenFOAM MSD slope error {err:.3e} exceeds {tol_sigma} sigma "
        f"({tol_sigma * sigma:.3e})")
    assert isotropy < 10.0 * math.sqrt(2.0 / n)


@pytest.mark.slow
def test_taylor_aris_openfoam_long_time():
    case = yaml.safe_load(TA_CASE.read_text())
    num = case["numerics"]
    n = 10000

    r = run_case(case, runner=RUNNER, n_particles=n, seed=0,
                 epsilon=float(num["epsilon"]), cycles=float(num["cycles"]),
                 workdir=REPO / "_runs")

    tau_r = r["meta"]["tau_r"]
    window = [x * tau_r for x in num["fit_window_tau_r"]]
    d_eff_err = METRICS["d_eff_relative"](
        r["var_x"], r["t"], window, r["D_eff_expected"])

    sigma = mc_error.sigma("slope_fit_variance_1d", n)
    tol_sigma = float(case["metrics"][0]["tol_sigma"])
    ks_floor = mc_error.ks_critical(n)
    ks_tol = float(case["metrics"][1]["tol_multiple_of_floor"])

    record = {
        "case": case["name"],
        "runner": RUNNER,
        "solver": r["meta"]["solver"],
        "note": "OpenFOAM 14 brownianTracer in a frozen analytic Poiseuille "
                "pipe (solver functions + subSolver incompressibleFluid); "
                "specular walls reflect the noise velocity as well as U. "
                "Long-time anchor + KS gate only (agreed scope).",
        "openfoam_version": r["meta"]["openfoam_version"],
        "cloud_library": r["meta"]["cloud_library"],
        "wall_scheme": r["meta"]["wall_scheme"],
        "reference": {
            "n_particles": n,
            "peclet": r["meta"]["peclet"],
            "stokes_number": r["meta"]["stokes_number"],
            "d_eff_over_d": r["D_eff_expected"] / r["D_expected"],
            "d_eff_relative": float(d_eff_err),
            "sigma_theory": sigma,
            "z_score": float(d_eff_err) / sigma,
            "ks_statistic": r["ks_statistic"],
            "ks_floor": ks_floor,
            "ks_over_floor": r["ks_statistic"] / ks_floor,
            "ks_history": r["ks_history"],
        },
        "error_laws_used": {
            "d_eff": "slope_fit_variance_1d = sqrt(6)/2",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(REPO),
    }
    TA_RESULTS.write_text(json.dumps(record, indent=2) + "\n")

    # KS first: a wrong wall treatment biases D_eff invisibly
    assert r["ks_statistic"] < ks_tol * ks_floor, (
        f"radial KS {r['ks_statistic']:.4f} exceeds {ks_tol} x floor "
        f"({ks_floor:.4f}) - wall treatment suspect")
    late = [s for t_, s in zip(r["ks_history"]["t_over_tau_r"],
                               r["ks_history"]["statistic"]) if t_ > 1.0]
    assert all(s < ks_tol * ks_floor for s in late), (
        f"KS history exceeds {ks_tol} x floor after tau_r: {late}")

    assert d_eff_err < tol_sigma * sigma, (
        f"D_eff error {d_eff_err:.3e} exceeds {tol_sigma} sigma "
        f"({tol_sigma * sigma:.3e})")
