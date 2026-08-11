"""The lattice-Boltzmann runner against the lattice-Boltzmann oracle.

The oracle's claims were established by algebra — symbolic dispersion
relations, 50-digit eigenvalue expansions. This test confronts them with an
actual collide-and-stream lattice, and it is the reference contact: when an
external LBM code (OpenLB) is measured against the oracle later, any
disagreement is localised against a lattice that is KNOWN to agree.

Three measurements, in increasing order of novelty:

  1. D = c_s^2 (tau - 1/2), including tau = 0.51 where the naive relation
     (the -1/2 dropped) is 51x too large. The run must land on 0.00333, and
     landing anywhere near 0.17 means the -1/2 is gone.
  2. The Ma^2 law: the first-order equilibrium's diffusivity is depleted by
     exactly u^2 (tau - 1/2); the second-order equilibrium cancels it at
     standard weights and OVERCORRECTS by +5 u^2 at OpenLB's thermal D2Q5
     weights (omega = 2/5).
  3. THE SLIP EXPERIMENT — the measurement the oracle did not have. The
     anti-bounce-back Dirichlet slip vanishes at Lambda = (tau-1/2)^2 = 3/16
     and matches the published formula at every tau under the matched
     convention (the paper's N is the half-width in lattice spacings; the
     unmatched comparison is off by exactly 4, which is how the convention
     was identified). The simple source scheme shifts the whole curve by
     exactly -S/2 — the missing half-step of the scalar redefinition.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.analytic import lattice_boltzmann as lb
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "lbm_scalar.yaml"
RESULTS_FILE = REPO / "results" / "lbm_scalar.json"


def test_lbm_scalar():
    case = yaml.safe_load(CASE_FILE.read_text())
    study = case["study"]
    tols = {m["name"]: float(m["tol"]) for m in case["metrics"]}
    record = {
        "case": case["name"],
        "runner": "lbm",
        "note": "pure-numpy collide-and-stream; the reference contact for the "
                "lattice_boltzmann oracle before any external LBM code",
    }

    # --- 1. The relaxation relation, sharpest at tau = 0.51 ----------------
    relax = []
    for tau in study["tau_sweep"]:
        r = run_case(case, runner="lbm", tau=float(tau))
        relax.append({
            "tau": float(tau),
            "d_measured": r["d_measured"],
            "d_exact": r["d_expected_standard"],
            "d_naive": lb.diffusivity_naive(float(tau), "D2Q9"),
            "relative": abs(r["d_measured"] / r["d_expected_standard"] - 1.0),
            "mass_drift": r["mass_drift"],
        })
    record["relaxation_relation"] = relax

    # --- 2. The Ma^2 law ---------------------------------------------------
    ma2 = []
    for u in study["u_sweep"]:
        r1 = run_case(case, runner="lbm", tau=1.0, u=float(u))
        r2 = run_case(case, runner="lbm", tau=1.0, u=float(u), equilibrium_order=2)
        ma2.append({
            "u": float(u),
            "mach2": float(u) ** 2 / r1["cs2"],
            "first_order_relative_to_depleted": abs(
                r1["d_measured"] / r1["d_expected_first_order"] - 1.0
            ),
            "first_order_depletion_measured": 1.0
            - r1["d_measured"] / r1["d_expected_standard"],
            "second_order_relative_to_standard": abs(
                r2["d_measured"] / r2["d_expected_standard"] - 1.0
            ),
        })
    record["ma2_law"] = ma2

    # OpenLB's thermal D2Q5 weights: the second-order term OVERCORRECTS.
    r5 = run_case(case, runner="lbm", lattice="D2Q5", omega=0.4, tau=1.0,
                  u=0.2, equilibrium_order=2)
    over_expected = r5["d_expected_standard"] * (
        1.0 + lb.d2q5_second_order_residual(0.2, 0.4)
    )
    record["d2q5_overcorrection"] = {
        "omega": 0.4,
        "cs2": r5["cs2"],
        "d_measured": r5["d_measured"],
        "d_with_residual": over_expected,
        "relative": abs(r5["d_measured"] / over_expected - 1.0),
        "overcorrection_percent": 100.0 * lb.d2q5_second_order_residual(0.2, 0.4),
    }

    # --- 3. The slip experiment -------------------------------------------
    slip_case = {**case, "experiment": "dirichlet_slip"}
    n_slip = int(study["slip_n"])
    sweep = []
    for tau in study["slip_tau_sweep"]:
        r = run_case(slip_case, runner="lbm", tau=float(tau), n_cells=n_slip)
        assert r["converged"], f"slip run at tau={tau} did not reach steady state"
        sweep.append({
            "tau": float(tau),
            "slip": r["slip_measured"],
            "published_matched": r["slip_published_form"],
            "uniformity": r["slip_uniformity"],
        })
    # Zero crossing MEASURED by root-finding on the lattice itself — no
    # model assumption. (A first version interpolated linearly on Lambda,
    # but slip ~ [Lambda - 3/16]/(tau - 1/2): the Delta_phi carries 1/D, so
    # the slip is NOT linear in Lambda and the two-point interpolation
    # missed the root by 5e-3. The bisection has no such model in it.)
    from scipy.optimize import brentq

    val = np.array([s["slip"] for s in sweep])

    def slip_at(tau):
        return run_case(slip_case, runner="lbm", tau=float(tau),
                        n_cells=n_slip)["slip_measured"]

    tau0 = float(brentq(slip_at, 0.9, 1.0, xtol=1e-7))
    record["slip"] = {
        "sweep": sweep,
        "zero_crossing_tau": tau0,
        "zero_crossing_exact": lb.zero_slip_tau(),
        "zero_crossing_lambda": lb.magic_lambda(tau0),
        "magic_lambda": 3.0 / 16.0,
        "convention_note": "published formula evaluated with N_paper = N/2 "
                           "(half-width); the unmatched form is off by "
                           "exactly 4 at every tau",
    }

    # The simple source scheme is the corrected one shifted by exactly -S/2.
    r_simple = run_case(slip_case, runner="lbm", tau=1.0, n_cells=n_slip,
                        source_scheme="simple")
    r_corr = run_case(slip_case, runner="lbm", tau=1.0, n_cells=n_slip)
    source = float(case["numerics"]["source"])
    record["source_scheme_shift"] = {
        "simple_minus_corrected": r_simple["slip_measured"] - r_corr["slip_measured"],
        "expected": -source / 2.0,
    }

    # N-independence at fixed lattice source IS the 1/N^2 law at fixed
    # Delta_phi: Delta_phi = S N^2 / (8 D) cancels the 1/N^2.
    n_scaling = []
    for n in study["slip_n_sweep"]:
        r = run_case(slip_case, runner="lbm", tau=1.5, n_cells=int(n))
        n_scaling.append({
            "n": int(n),
            "slip": r["slip_measured"],
            "slip_over_delta_phi": r["slip_measured"] / r["delta_phi"],
        })
    record["slip_n_scaling"] = {
        "runs": n_scaling,
        "note": "slip/Delta_phi must fall as 1/N^2; raw slip is N-independent "
                "at fixed lattice source",
    }

    record["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["git_sha"] = git_sha(REPO)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    # --- assertions --------------------------------------------------------
    # 1. The relaxation relation, and the naive form decisively excluded.
    for row in relax:
        assert row["relative"] < tols["d_relative"], (
            f"D at tau={row['tau']}: relative {row['relative']:.2e}"
        )
        assert row["mass_drift"] < 1e-12
    worst = next(r for r in relax if r["tau"] == 0.51)
    assert worst["d_naive"] / worst["d_measured"] > 50.0, (
        "at tau = 0.51 the naive relation must be ~51x the measurement; if "
        "this fails the -1/2 discrimination has been lost"
    )

    # 2. Ma^2: depletion matches -u^2/c_s^2 and the second order cancels it.
    for row in ma2:
        assert row["first_order_relative_to_depleted"] < tols["d_relative"]
        assert row["second_order_relative_to_standard"] < tols["d_relative"]
        if row["u"] > 0:
            assert row["first_order_depletion_measured"] == pytest.approx(
                row["mach2"], rel=1e-4
            ), "the measured depletion IS Ma^2 — the coefficient is exactly -1"
    assert record["d2q5_overcorrection"]["relative"] < tols["d_relative"]
    assert record["d2q5_overcorrection"]["overcorrection_percent"] == pytest.approx(
        20.0, rel=1e-9
    )

    # 3. The slip: sign flip, zero at the magic tau, published prefactor
    #    under the matched convention, uniform offset.
    assert val[0] < 0.0 < val[-1], "the slip must change sign across the sweep"
    assert abs(tau0 - lb.zero_slip_tau()) < 1e-5, (
        f"slip zero at tau = {tau0:.7f}, exact {lb.zero_slip_tau():.7f} — "
        f"the bisected zero should land on the magic tau to the steady floor"
    )
    for s in sweep:
        if abs(s["published_matched"]) > 1e-9:
            assert abs(s["slip"] / s["published_matched"] - 1.0) < tols[
                "slip_vs_published"
            ], f"slip at tau={s['tau']} disagrees with the matched formula"
        assert s["uniformity"] < 1e-10, (
            "the slip must be UNIFORM; structure in the deviation means the "
            "wall or source scheme differs from the analysed one"
        )

    # The simple-source shift is exactly -S/2 (the scalar redefinition).
    shift = record["source_scheme_shift"]
    assert shift["simple_minus_corrected"] == pytest.approx(
        shift["expected"], rel=1e-9
    )

    # slip/Delta_phi falls as 1/N^2 (the published law); raw slip is
    # N-independent at fixed source.
    ratios = [row["slip_over_delta_phi"] for row in n_scaling]
    assert ratios[0] / ratios[1] == pytest.approx(4.0, rel=1e-6)
    assert ratios[1] / ratios[2] == pytest.approx(4.0, rel=1e-6)
    raw = [row["slip"] for row in n_scaling]
    assert raw[0] == pytest.approx(raw[2], rel=1e-9)
