"""Taylor-Aris dispersion — the LAST exact oracle in this programme.

Four independent exact anchors, three new in kind:
  1. D_eff = D + a^2 U^2/(48 D) from the long-time variance slope.
  2. A DIFFERENT POWER OF t at short times: sigma_x^2 - 2Dt -> (U^2/3) t^2,
     which a code that lands D_eff right by accident still misses.
  3. P(r) = 2r/a^2 at all times, with and without flow. THE GATE.
  4. The SHAPE of D_eff(R), via R* from a two-parameter fit, not a point.

Every Monte Carlo tolerance is sigma-scaled with the law for ITS estimator:
D_eff is a fitted 1-D variance (sqrt(6)/2 / sqrt(N)), not a single-time
variance and not a fitted 3-D MSD. See betaflow/metrics/mc_error.py.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.analytic import taylor_aris as ta
from betaflow.metrics import METRICS, mc_error
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "taylor_aris.yaml"
RESULTS_FILE = REPO / "results" / "taylor_aris.json"

# Declared from the saturation correction, not fitted: over the ballistic
# window the residual is Var(u) t^2 (1 - (beta_1^2/3) t/tau_r + ...), which
# biases the fitted prefactor a few percent low.
PREFACTOR_BAND = (0.90, 1.05)


@pytest.mark.slow
def test_taylor_aris():
    case = yaml.safe_load(CASE_FILE.read_text())
    num = case["numerics"]
    study = case["study"]
    phys = case["physical"]
    a = float(phys["vessel_radius"])
    u_mean = float(phys["mean_velocity"])
    eps = float(num["epsilon"])
    cycles = float(num["cycles"])

    oracle_checks = ta.verify_limits()

    def go(n, seed=0, **kw):
        return run_case(case, runner="langevin", n_particles=n, seed=seed,
                        epsilon=kw.pop("epsilon", eps),
                        cycles=kw.pop("cycles", cycles), **kw)

    def d_eff_err(r):
        w = [x * r["meta"]["tau_r"] for x in num["fit_window_tau_r"]]
        return METRICS["d_eff_relative"](r["var_x"], r["t"], w, r["D_eff_expected"])

    # --- 1. Particle-count study; its finest run IS the reference ----------
    counts = [int(n) for n in study["particle_counts"]]
    count_runs = []
    ref = None
    for n in counts:
        r = go(n, seed=0)
        err = d_eff_err(r)
        count_runs.append({
            "n_particles": n,
            "d_eff_relative": err,
            "sigma_theory": mc_error.sigma("slope_fit_variance_1d", n),
            "z_score": err / mc_error.sigma("slope_fit_variance_1d", n),
            "ks_statistic": r["ks_statistic"],
            "ks_floor": mc_error.ks_critical(n),
        })
        if n == int(study["reference_particles"]):
            ref = r
    assert ref is not None

    # --- 2. Epsilon calibrated BY THE GATE, not assumed --------------------
    n_cal = int(study["epsilon_calibration_particles"])
    eps_runs = []
    for e in [float(x) for x in num["epsilon_sweep"]]:
        r = go(n_cal, seed=1, epsilon=e)
        eps_runs.append({
            "epsilon": e,
            "n_steps": r["meta"]["n_steps"],
            "ks_statistic": r["ks_statistic"],
            "ks_floor": mc_error.ks_critical(n_cal),
            "ks_over_floor": r["ks_statistic"] / mc_error.ks_critical(n_cal),
            "d_eff_relative": d_eff_err(r),
        })

    # --- 3. Ballistic regime, on a dedicated short run ---------------------
    short = go(50000, seed=2, epsilon=float(num["short_time_epsilon"]),
               cycles=float(num["short_time_cycles"]))
    sw = [x * short["meta"]["tau_r"] for x in num["short_time_window_tau_r"]]
    exponent, prefactor = METRICS["short_time_exponent"](
        short["var_x"], short["t"], sw, short["D_expected"]
    )
    short_time = {
        "exponent": exponent,
        "expected_exponent_band": [float(x) for x in num["short_time_exponent_band"]],
        "prefactor": prefactor,
        "prefactor_over_u2_3": prefactor / ta.velocity_variance(u_mean),
        "n_steps": short["meta"]["n_steps"],
    }

    # --- 4. Radius sweep: R* from a two-parameter fit to the WHOLE curve ---
    radii = [float(x) * 1e-9 for x in study["radius_sweep_nm"]]
    n_sweep = int(study["radius_sweep_particles"])
    sweep = []
    for i, r_p in enumerate(radii):
        r = go(n_sweep, seed=10 + i, particle_radius=r_p)
        w = [x * r["meta"]["tau_r"] for x in num["fit_window_tau_r"]]
        sweep.append({
            "particle_radius_nm": r_p * 1e9,
            "peclet": r["meta"]["peclet"],
            "d_eff_measured": METRICS["d_eff_slope"](r["var_x"], r["t"], w),
            "d_eff_exact": r["D_eff_expected"],
            "ks_statistic": r["ks_statistic"],
        })
    # D_eff(R) = A/R + B R.  R* = sqrt(A/B). The minimum is never located by
    # scanning: the curve is flat there and a two-parameter fit uses all K
    # points, so the error in R* falls as ~1/sqrt(K).
    rr = np.array([s["particle_radius_nm"] * 1e-9 for s in sweep])
    dd = np.array([s["d_eff_measured"] for s in sweep])
    design = np.column_stack((1.0 / rr, rr))
    (coef_a, coef_b), *_ = np.linalg.lstsq(design, dd, rcond=None)
    r_star = math.sqrt(coef_a / coef_b)
    r_cr = ta.critical_radius(float(phys["temperature"]),
                              float(phys["dynamic_viscosity"]), a, u_mean)

    record = {
        "case": case["name"],
        "runner": "langevin",
        "note": "LAST exact oracle; analytic Poiseuille field, no CFD solver",
        "oracle_self_verification": oracle_checks,
        "asymptotic_onset_tau_r": ta.asymptotic_onset(),
        "bessel_j1_first_root": ta.BESSEL_J1_FIRST_ROOT,
        "reference": {
            "n_particles": ref["meta"]["n_particles"],
            "peclet": ref["meta"]["peclet"],
            "stokes_number": ref["meta"]["stokes_number"],
            "d_eff_over_d": ref["D_eff_expected"] / ref["D_expected"],
            "d_eff_relative": d_eff_err(ref),
            "ks_history": ref["ks_history"],
            "wall_scheme": ref["meta"]["wall_scheme"],
        },
        "particle_count_study": count_runs,
        "epsilon_calibration": eps_runs,
        "short_time": short_time,
        "radius_sweep": {
            "runs": sweep,
            "fit_A": float(coef_a),
            "fit_B": float(coef_b),
            "r_star_nm": r_star * 1e9,
            "r_cr_nm": r_cr * 1e9,
            "r_star_over_r_cr": r_star / r_cr,
        },
        "error_laws_used": {
            "d_eff": "slope_fit_variance_1d = sqrt(6)/2",
            "note": "NOT sqrt(2) (single-time variance) and NOT 0.7071 "
                    "(fitted 3-D MSD); four laws differ by up to 2x",
        },
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(REPO),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    # --- assertions --------------------------------------------------------
    tol_sigma = float(case["metrics"][0]["tol_sigma"])

    # THE GATE first: P(r) = 2r/a^2 is exact, independent of the dispersion
    # physics, and the only check that catches a wrong wall BC — which would
    # bias D_eff invisibly through the velocity sampling.
    ks_limit = float(case["metrics"][1]["tol_multiple_of_floor"])
    for entry in count_runs:
        assert entry["ks_statistic"] < ks_limit * entry["ks_floor"], (
            f"radial distribution departs from 2r/a^2 at N={entry['n_particles']}: "
            f"KS {entry['ks_statistic']:.3e} vs floor {entry['ks_floor']:.3e} — "
            f"suspect the wall scheme, which biases D_eff invisibly"
        )
    for t_over_tau, stat in zip(ref["ks_history"]["t_over_tau_r"],
                                ref["ks_history"]["statistic"]):
        assert stat < ks_limit * mc_error.ks_critical(ref["meta"]["n_particles"]), (
            f"radial distribution drifts at t/tau_r = {t_over_tau:.2f}: KS {stat:.3e}"
        )

    # D_eff, sigma-scaled with the law for a fitted 1-D variance.
    for entry in count_runs:
        assert entry["z_score"] < tol_sigma, (
            f"D_eff error {entry['d_eff_relative']:.3e} is "
            f"{entry['z_score']:.2f} sigma at N={entry['n_particles']} "
            f"(sigma = {entry['sigma_theory']:.3e}, limit {tol_sigma})"
        )
    # Monte Carlo rate over the ensemble axis.
    zs = np.array([e["d_eff_relative"] for e in count_runs])
    ns = np.array([e["n_particles"] for e in count_runs], dtype=float)
    mc_exponent = float(-np.polyfit(np.log(ns), np.log(zs), 1)[0])

    # Epsilon: the gate must sit at its floor, which VERIFIES the timestep
    # rather than assuming it. If KS rose above floor and fell with epsilon,
    # a real wall-handling error would be being resolved.
    for entry in eps_runs:
        assert entry["ks_over_floor"] < ks_limit, (
            f"at epsilon={entry['epsilon']} the radial KS is "
            f"{entry['ks_over_floor']:.2f}x its floor — wall handling is "
            f"resolving a real error, so epsilon is not yet calibrated"
        )

    # Ballistic regime: a DIFFERENT power of t.
    lo, hi = short_time["expected_exponent_band"]
    assert lo < exponent < hi, (
        f"short-time exponent {exponent:.4f} outside {lo}-{hi}; the shear "
        f"sampling regime should grow as t^2, not as the t^1 of dispersion"
    )
    assert PREFACTOR_BAND[0] < short_time["prefactor_over_u2_3"] < PREFACTOR_BAND[1], (
        f"short-time prefactor is {short_time['prefactor_over_u2_3']:.4f} of "
        f"U^2/3, outside {PREFACTOR_BAND}"
    )

    # The SHAPE of D_eff(R): R* from the whole curve must recover Eq. 18.
    assert 0.8 < r_star / r_cr < 1.25, (
        f"R* from the two-parameter fit is {r_star*1e9:.2f} nm against "
        f"Eq. 18's {r_cr*1e9:.2f} nm (ratio {r_star/r_cr:.3f})"
    )
    # ... and the curve must actually turn: falling below R_cr, rising above.
    below = [s for s in sweep if s["particle_radius_nm"] * 1e-9 < 0.5 * r_cr]
    above = [s for s in sweep if s["particle_radius_nm"] * 1e-9 > 2.0 * r_cr]
    assert below[0]["d_eff_measured"] > below[-1]["d_eff_measured"], (
        "D_eff must FALL with particle size below R_cr (diffusion-dominated)"
    )
    assert above[0]["d_eff_measured"] < above[-1]["d_eff_measured"], (
        "D_eff must RISE with particle size above R_cr (convection-dominated)"
    )

    record["particle_count_mc_exponent"] = mc_exponent
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")
