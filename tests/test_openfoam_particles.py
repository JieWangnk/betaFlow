"""OpenFOAM 14 particle runner vs the exact Brownian/Taylor-Aris analytic references.

The three-way check this closes: the analytic references are exact, runners/langevin.py is
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


@pytest.mark.slow
def test_mc_channel_openfoam():
    """The CIR through the OpenFOAM Lagrangian tracker — both rungs.

    D = 0 IS the replication of Hofmann et al. 2024's published model class
    (their model has no diffusion; their MPPIC setup is not reproducible in
    stock OF14 and their DMPPIC source is deleted — recorded below). D > 0
    exceeds their model and must reproduce the two-regime tail the Langevin
    leg measured. Tolerances: binomial laws, with a wider RMSE band than the
    Langevin leg because the OF particles sample the cellPoint INTERPOLANT
    of the parabola — a one-sign bias (speeds low, arrivals late) predicted
    in runners/openfoam_particles.py before the first run and recorded here.
    """
    import yaml as _yaml
    from betaflow.analytic import channel_impulse as ci
    from betaflow.metrics import mc_error

    case = _yaml.safe_load(
        (REPO / "betaflow" / "cases" / "mc_channel.yaml").read_text())
    out_file = REPO / "results" / "mc_channel_openfoam.json"

    # ---- Replication rung: D = 0, exact kinematics up to interpolation ----
    n0 = 20000
    r0 = run_case(case, runner="openfoam_particles", n_particles=n0, seed=0,
                  diffusivity=0.0, time_horizon_over_t2=6.0, writes=120)
    ks_floor0 = mc_error.ks_critical(n0, alpha=0.05)
    assert r0["ks_statistic"] < 1.5 * ks_floor0

    v_mean = float(case["physical"]["mean_velocity"])
    c_x = float(case["receiver"]["axial_length"])
    replication = []
    for rec in r0["receivers"]:
        t, cm, co = rec["t"], rec["cir_measured"], rec["cir_reference"]
        # Pre-onset EXACTLY zero: interpolation can only delay arrivals.
        pre = cm[t < rec["t1"]]
        assert pre.size > 0 and float(np.max(pre)) == 0.0, (
            f"dbar={rec['dbar']*1e6:.0f}um: arrival before t1 at D = 0 — "
            f"impossible under a speed field bounded by 2V")
        rmse = float(np.sqrt(np.mean((cm - co) ** 2)))
        floor = mc_error.binomial_rmse_floor(co, n0)
        # Band wider than the Langevin leg's (0.3, 2.0): the interpolation
        # bias adds a deterministic component the binomial law does not
        # cover. The measured ratio is recorded; a ratio above 3 would mean
        # the bias dominates and the mesh needs stating, not the tolerance.
        assert 0.3 < rmse / floor < 3.0, (
            f"dbar={rec['dbar']*1e6:.0f}um: RMSE/floor {rmse/floor:.2f}")
        peak_gap = float(np.max(cm)) - float(np.max(co))
        sig_pk = mc_error.binomial_sigma(rec["peak_value"], n0)
        assert abs(peak_gap) < 5.0 * sig_pk, (
            f"dbar={rec['dbar']*1e6:.0f}um: sampled peak gap "
            f"{peak_gap/sig_pk:+.1f} sigma")
        sel = t >= rec["t2"]
        mass_m = float(np.trapezoid(cm[sel], t[sel]))
        mass_r = float(np.trapezoid(co[sel], t[sel]))
        bound = mc_error.binomial_integral_sigma_bound(t[sel], co[sel], n0)
        assert abs(mass_m - mass_r) < 4.0 * bound
        # The one-sign bias, measured: CIR centroid shift in units of t2.
        centroid_shift = (float(np.trapezoid(t * (cm - co), t))
                          / float(np.trapezoid(co, t)) / rec["t2"])
        replication.append({
            "dbar_um": rec["dbar"] * 1e6,
            "rmse_over_binomial_floor": rmse / floor,
            "peak_gap_sigma": peak_gap / sig_pk,
            "tail_mass_gap_over_bound": (mass_m - mass_r) / bound,
            "centroid_shift_t2_units": centroid_shift,
            "pre_onset_max": float(np.max(pre)),
        })

    # ---- Beyond-their-model rung: D > 0, the two-regime tail ----
    # writes=400: the near receiver's pre-onset window [0.85 t1, t1) is only
    # 5 ms wide, so the write cadence must resolve it — at 160 writes the
    # first run had NO sample in that window and the check failed on sample
    # count, not physics.
    n1 = 10000
    r1 = run_case(case, runner="openfoam_particles", n_particles=n1, seed=1,
                  time_horizon_over_t2=6.5, writes=400)
    ks_floor1 = mc_error.ks_critical(n1, alpha=0.05)
    assert r1["ks_statistic"] < 1.5 * ks_floor1, (
        f"radial KS {r1['ks_statistic']:.4f} above 1.5x floor — the wall "
        f"scheme is suspect and every CIR number below it")

    departure = []
    for rec in r1["receivers"]:
        t, cm = rec["t"], rec["cir_measured"]
        near = cm[(t >= 0.85 * rec["t1"]) & (t < rec["t1"])]
        assert near.size > 0 and float(np.max(near)) > 0.0, (
            f"dbar={rec['dbar']*1e6:.0f}um: no pre-onset arrivals at D > 0")
        peak_ratio = float(np.max(cm)) / rec["peak_value"]
        assert peak_ratio < 1.0
        departure.append({
            "dbar_um": rec["dbar"] * 1e6,
            "peak_measured_over_reference": peak_ratio,
            "pre_onset_max": float(np.max(near)),
        })

    mid = next(r for r in r1["receivers"] if abs(r["dbar"] - 750e-6) < 1e-9)
    t, cm, co, t2 = mid["t"], mid["cir_measured"], mid["cir_reference"], mid["t2"]
    sel = (t >= 3.0 * t2) & (t <= 5.0 * t2)
    excess = float(np.trapezoid(cm[sel] - co[sel], t[sel]))
    bound = mc_error.binomial_integral_sigma_bound(t[sel], co[sel], n1)
    assert excess > 2.0 * bound, (
        f"tail enhancement {excess:.2e} below 2x bound {bound:.2e} — the "
        f"Langevin leg measured the reservoir-pumping regime here")
    k12 = int(np.argmin(np.abs(t - 12.0 * t2)))
    assert cm[k12] < 0.2 * co[k12], (
        f"CIR at 12 t2 is {cm[k12]:.2e} vs reference {co[k12]:.2e} — "
        f"the tail should have terminated")

    record = {
        "case": "mc_channel",
        "solver_leg": "openfoam_particles (brownianTracer cloud)",
        "replication_claim": (
            "D = 0 replicates Hofmann et al. 2024's published model class "
            "(Lagrangian particles in flow, no diffusion). Method-class, "
            "not MPPIC-faithful: stock OF14 registers the Brownian force "
            "only for thermo-family clouds MPPIC cannot construct, and the "
            "authors' DMPPIC solver source is deleted from GitHub with no "
            "archived copy (both verified 2026-08)."
        ),
        "interpolation_bias_prediction": (
            "written before the run: cellPoint interpolation of the concave "
            "parabola biases speeds low, one sign — arrivals late, never "
            "early"
        ),
        "replication_rung_D0": {
            "n_particles": n0,
            "ks_statistic": r0["ks_statistic"],
            "ks_floor": ks_floor0,
            "receivers": replication,
        },
        "departure_rung": {
            "n_particles": n1,
            "ks_statistic": r1["ks_statistic"],
            "ks_floor": ks_floor1,
            "receivers": departure,
            "middle_receiver_tail": {
                "enhancement_excess_3_to_5_t2": excess,
                "enhancement_sigma_bound": bound,
                "cir_at_12_t2_measured": float(cm[k12]),
                "cir_at_12_t2_reference": float(co[k12]),
            },
        },
        "meta_D0": r0["meta"],
        "meta_departure": r1["meta"],
        "git_sha": git_sha(REPO),
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    out_file.write_text(json.dumps(record, indent=2) + "\n")
