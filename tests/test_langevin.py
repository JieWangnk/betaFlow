"""Free Brownian motion — physics, and the first test of the runner abstraction.

DOUBLE PURPOSE. The physics is Stokes-Einstein diffusion validated against
MSD = 6 D t. The architecture question is whether betaflow's central design
claim — nothing above runners/ knows about the solver — survives contact with
a runner that is not OpenFOAM. Until this case, every runner WAS OpenFOAM, so
the claim was asserted rather than tested.

The convergence axis here is PARTICLE COUNT, not resolution. The error is
statistical and falls as 1/sqrt(N), so the expected exponent is 0.5 — a Monte
Carlo rate, not the 2 of a discretisation study. Any refinement machinery that
assumes p ~= 2 would be wrong here, which is itself part of the test.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.analytic import brownian
from betaflow.metrics import METRICS
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "langevin_free.yaml"
RESULTS_FILE = REPO / "results" / "langevin_free.json"

# Monte Carlo, so the exponent is 1/2. Band declared a priori from that rate,
# not fitted: with 8 replicas the estimate of a standard error is itself noisy
# at the ~25% level, which maps to roughly +/-0.15 on a two-decade log-log fit.
MC_EXPONENT_BAND = (0.35, 0.65)


def _slope_error(case, n_particles, seed, **kwargs):
    r = run_case(
        case, runner="langevin", n_particles=n_particles, seed=seed, **kwargs
    )
    return METRICS["msd_slope_relative"](r["msd"], r["t"], r["D_expected"]), r


def test_langevin_free():
    case = yaml.safe_load(CASE_FILE.read_text())
    study = case["study"]
    counts = [int(n) for n in study["particle_counts"]]
    replicas = int(study["replicas"])
    n_ref = int(study["reference_particles"])
    spec = case["metrics"][0]
    sigma_coeff = float(spec["sigma_coefficient"])
    tol_sigma = float(spec["tol_sigma"])

    def sigma_of(n):
        """Predicted standard error of the MSD-slope estimate at ensemble N."""
        return sigma_coeff / math.sqrt(n)

    tol = tol_sigma * sigma_of(n_ref)

    # 0. Verify the verifier: the noise amplitude and the Stokes drag must
    #    share one friction coefficient, or fluctuation-dissipation is broken.
    fdt = brownian.verify_fluctuation_dissipation(
        float(case["physical"]["temperature"]),
        float(case["physical"]["dynamic_viscosity"]),
        float(case["physical"]["particle_radius"]),
    )

    # 1. Reference run at the case's stated ensemble size.
    err_ref, r_ref = _slope_error(case, n_ref, seed=0)
    meta = r_ref["meta"]

    # 2. Statistical convergence over PARTICLE COUNT. Averaging |error| over
    #    replicas estimates the standard error; a single seed per N would be
    #    far too noisy to fit an exponent to.
    sweep = []
    for n in counts:
        errs = [_slope_error(case, n, seed=s)[0] for s in range(replicas)]
        sweep.append({
            "n_particles": n,
            "rms_slope_error": float(np.sqrt(np.mean(np.square(errs)))),
            "errors": [float(e) for e in errs],
        })
    logs_n = np.log(np.array([s["n_particles"] for s in sweep], dtype=float))
    logs_e = np.log(np.array([s["rms_slope_error"] for s in sweep]))
    mc_exponent = float(-np.polyfit(logs_n, logs_e, 1)[0])

    # 3. Timestep independence. Euler-Maruyama is EXACT for free diffusion,
    #    so halving dt must change nothing beyond the RNG draw.
    dt_study = []
    for n_steps in (50, 100, 200):
        e, rr = _slope_error(case, n_ref, seed=7, n_steps=n_steps)
        dt_study.append({"n_steps": n_steps, "dt": rr["meta"]["dt"],
                         "slope_error": float(e)})

    # 4. Isotropy: the three per-component MSDs must agree within statistics.
    comp = r_ref["msd_components"][-1]
    isotropy = float((comp.max() - comp.min()) / comp.mean())

    record = {
        "case": case["name"],
        "runner": "langevin",
        "solver": meta["solver"],
        "note": "NO CFD solver — first non-OpenFOAM runner",
        "fluctuation_dissipation": fdt,
        "diffusivity": meta["diffusivity"],
        "stokes_number": meta["stokes_number"],
        "rms_displacement_m": meta["rms_displacement"],
        "reference": {
            "n_particles": n_ref, "slope_error": err_ref,
            "sigma_theory": sigma_of(n_ref), "tol_sigma": tol_sigma,
            "tol_absolute": tol,
            "z_score": err_ref / sigma_of(n_ref),
        },
        "particle_count_study": {"runs": sweep, "mc_exponent": mc_exponent,
                                 "expected_exponent": 0.5},
        "timestep_study": dt_study,
        "isotropy_spread": isotropy,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "git_sha": git_sha(REPO),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    # --- assertions --------------------------------------------------------
    # 1. MSD slope recovers 6 D at the stated ensemble size.
    assert err_ref < tol, (
        f"MSD slope error {err_ref:.3e} exceeds {tol_sigma:.0f} sigma "
        f"({tol:.3e}) at N={n_ref}; sigma = {sigma_of(n_ref):.3e}"
    )
    # ... and at every N in the sweep, the error is consistent with statistics
    # rather than bias: it must fall as N grows.
    rms = [s["rms_slope_error"] for s in sweep]
    assert all(c > f for c, f in zip(rms, rms[1:])), (
        f"statistical error should fall monotonically with N: {rms}"
    )

    # 2. Monte Carlo rate, NOT a discretisation rate.
    assert MC_EXPONENT_BAND[0] < mc_exponent < MC_EXPONENT_BAND[1], (
        f"error should scale as 1/sqrt(N) (exponent 0.5); measured "
        f"{mc_exponent:.3f}. This is a Monte Carlo rate — a value near 2 "
        f"would mean the study is measuring discretisation, not statistics."
    )

    # 3. Timestep independence: Euler-Maruyama is exact for free diffusion.
    #    All three dt share a seed, so they differ only through the number of
    #    random draws; the spread must stay at the statistical level.
    dt_errors = [d["slope_error"] for d in dt_study]
    assert max(dt_errors) < tol_sigma * sigma_of(n_ref), (
        f"halving dt must not degrade the answer (the scheme is exact for "
        f"free diffusion); errors {dt_errors}"
    )

    # 4. Isotropy: any real anisotropy is an RNG or indexing bug, not physics.
    #    Per-component MSD has relative statistical spread ~sqrt(2/N).
    assert isotropy < 10.0 * math.sqrt(2.0 / n_ref), (
        f"per-component MSD spread {isotropy:.3e} is too large to be "
        f"statistical at N={n_ref} — suspect the RNG or component indexing"
    )


@pytest.mark.slow
def test_langevin_error_distribution():
    """Does the sampler's error DISTRIBUTION match the statistical model?

    The Monte Carlo analogue of an order-of-accuracy test. Checking one draw
    against a threshold tests whether N is large enough; checking many draws
    against the predicted sigma tests the error model itself — and it is the
    only test that catches a sampler whose mean is right and whose variance is
    wrong (correlated draws, a short-period RNG, a reused stream).
    """
    case = yaml.safe_load(CASE_FILE.read_text())
    study = case["study"]
    n = int(study["reference_particles"])
    seeds = int(study["distribution_seeds"])
    sigma = float(case["metrics"][0]["sigma_coefficient"]) / math.sqrt(n)

    z = np.array([
        _slope_error(case, n, seed=1000 + s)[0] / sigma for s in range(seeds)
    ])
    within1 = float(np.mean(z < 1.0))
    within2 = float(np.mean(z < 2.0))
    out = {
        "n_particles": n, "seeds": seeds, "sigma_theory": sigma,
        "abs_z": [float(v) for v in z],
        "fraction_within_1sigma": within1, "fraction_within_2sigma": within2,
        "rms_z": float(np.sqrt(np.mean(z**2))),
    }
    path = RESULTS_FILE.with_name("langevin_error_distribution.json")
    path.write_text(json.dumps(out, indent=2) + "\n")

    # |z| for a standard normal: P(<1) = 0.683, P(<2) = 0.954. With 20 seeds
    # the binomial sd on those fractions is ~0.10 and ~0.05, so these bands
    # are ~3 sd wide — declared from the model, not fitted.
    assert 0.38 < within1 < 0.95, f"fraction within 1 sigma = {within1}"
    assert 0.75 < within2 <= 1.0, f"fraction within 2 sigma = {within2}"
    # The measured spread must match the predicted sigma, not merely be small:
    # this is what a variance-wrong sampler fails.
    assert 0.6 < out["rms_z"] < 1.6, (
        f"RMS z = {out['rms_z']:.3f}; the measured spread disagrees with the "
        f"predicted sigma = {sigma:.3e}, so the error MODEL is wrong even if "
        f"the mean is right"
    )
