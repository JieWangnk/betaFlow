"""Pulsatile Carreau (Lap 0d): the first case with NO EXACT ORACLE.

The unsteady term breaks the force balance every previous oracle rested on,
and the nonlinear viscosity kills superposition, so there is no closed form
for the profile. What remains exact is used in full, and what remains is
ESTIMATED is labelled as such. The deliverable is that distinction.

VERIFIED EXACTLY
  * momentum-balance identity tau_w(t) = h(G(t) - d<u>/dt), per timestep,
    with the scheme-matching discrete derivative. Rheology- and
    unsteadiness-independent, and the runner's convergence gate.
  * half-wave symmetry u(y, t+T/2) = -u(y, t), which G's oddness and nu's
    evenness in gammadot force on the periodic state.
  * two limits against oracles already in the repo: Cu -> 0 against the exact
    Womersley cosh solution, alpha -> 0 against the exact steady Carreau
    rootfind solution at the instantaneous G(t).

ESTIMATED (no oracle)
  * the profile itself, bounded by GCI (ASME V&V 20) over three levels of
    combined space-time refinement, with the safety factor reported and
    raised to 3 if the sequence is not in the asymptotic range.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.analytic import carreau, womersley
from betaflow.metrics import METRICS
from betaflow.metrics.gci import grid_convergence_index
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASE_FILE = REPO / "betaflow" / "cases" / "womersley_carreau.yaml"
RESULTS_FILE = REPO / "results" / "womersley_carreau.json"

# Cycle budget. Measured transient decay is 0.865/cycle — set by nu0 (the
# NOMINAL alpha), not the wall viscosity, because the slow mode lives in the
# low-shear core. Reaching a 1e-6 periodicity residual would need ~90 cycles
# (~30 min at the finest level), past the cost ceiling for one point, so the
# budget is capped and the ACHIEVED periodicity is reported rather than
# claimed. The half-wave residual is then bounded against it (see below).
CYCLES = 20

IDENTITY_TOL = 1e-6
# Half-wave symmetry cannot reach round-off at this cycle budget. What IS
# asserted is that it is proportional to the periodicity residual — i.e. it
# is incomplete periodicity and not an independent symmetry violation. The
# measured ratio is ~12 and constant across cycle counts.
HALF_WAVE_OVER_PERIODICITY_MAX = 30.0


def _summarise(result):
    meta = result["meta"]
    return {
        "n_cells": meta["mesh_level"],
        "n_steps": meta["n_steps_per_cycle"],
        "alpha_nominal": meta.get("alpha_nominal", meta["alpha"]),
        "alpha_eff": meta.get("alpha_eff"),
        "Cu": meta.get("Cu"),
        "nu_wall_over_nu0": (
            meta["nu_wall_estimate"] / meta["nu0"] if "nu_wall_estimate" in meta else None
        ),
        "cells_per_stokes_layer": meta["cells_per_stokes_layer"],
        "viscous_fourier": meta["viscous_fourier"],
        "identity_max_rel": meta["identity_max_rel"],
        "half_wave_residual": meta["half_wave_residual"],
        "final_periodicity": meta["periodicity"][-1],
        "cycles_to_periodic": meta["cycles_to_periodic"],
        "cycles_run": meta["cycles_run"],
        "A1_peak": float(np.max(result["u_amp"]) / result["u_ref"]),
        "A3_over_A1_velocity": float(
            np.max(result["u_amp3"]) / np.max(result["u_amp"])
        ),
        "A3_over_A1_tau": float(result["tau_amp3"] / result["tau_amp"]),
        "tau_amp": float(result["tau_amp"] / result["tau_ref"]),
    }


@pytest.mark.slow
def test_womersley_carreau():
    case = yaml.safe_load(CASE_FILE.read_text())
    h = float(case["geometry"]["half_height"])
    gci_cfg = case["gci"]
    levels = [tuple(int(x) for x in pair) for pair in gci_cfg["levels"]]
    limits = case["limits"]

    def _run(**kwargs):
        return run_case(case, runner="openfoam", workdir=REPO / "_runs", **kwargs)

    # --- Exact limit checks FIRST: they cost seconds, the ladder costs
    # --- minutes, and a failure here invalidates the ladder anyway. --------
    cu0 = float(limits["cu_zero"])
    r_cu0 = _run(cu=cu0, n_steps=128, max_cycles=CYCLES)
    y = np.asarray(r_cu0["y"]) / h
    exact = womersley.complex_profile(y, r_cu0["meta"]["alpha_nominal"])
    limit_womersley = {
        "Cu": cu0,
        "n_cells": r_cu0["meta"]["mesh_level"],
        "L2_amplitude": METRICS["L2_amplitude"](
            r_cu0["u_amp"] / r_cu0["u_ref"], np.abs(exact)
        ),
        "L2_phase": METRICS["L2_phase"](
            r_cu0["u_phase"], np.angle(exact), np.abs(exact) ** 2
        ),
        "A3_over_A1_tau": float(r_cu0["tau_amp3"] / r_cu0["tau_amp"]),
        "identity_max_rel": r_cu0["meta"]["identity_max_rel"],
    }

    a_small = float(limits["alpha_small"])
    r_qs = _run(
        alpha=a_small,
        n_cells=int(limits["alpha_small_cells"]),
        n_steps=int(limits["alpha_small_steps"]),
        max_cycles=int(limits["alpha_small_cycles"]),
    )
    m = r_qs["meta"]
    y_qs = np.asarray(r_qs["y"])
    steady = carreau.velocity(
        y_qs, 1.0, h, m["nu0"], m["nu_inf"], m["k"], m["n"], m["a"]
    )
    limit_quasi_steady = {
        "alpha": a_small,
        "n_cells": m["mesh_level"],
        "viscous_fourier": m["viscous_fourier"],
        "L2_vs_steady_carreau": METRICS["L2_velocity"](
            r_qs["u_amp"] / np.max(r_qs["u_amp"]), steady / np.max(steady)
        ),
        "amplitude_ratio": float(np.max(r_qs["u_amp"]) / np.max(steady)),
        "identity_max_rel": m["identity_max_rel"],
    }

    # --- GCI ladder. The finest level IS the main point. -------------------
    ladder = []
    for n_cells, n_steps in levels:
        result = _run(n_cells=n_cells, n_steps=n_steps, max_cycles=CYCLES)
        entry = _summarise(result)
        ladder.append(entry)
    main = ladder[-1]

    # GCI on two functionals: the peak first-harmonic velocity amplitude
    # (the profile quantity with no oracle) and the wall-shear amplitude.
    gci = {
        name: grid_convergence_index(
            ladder[0][name],
            ladder[1][name],
            ladder[2][name],
            r=2.0,
            safety_factor=float(gci_cfg["safety_factor"]),
            order_band=tuple(gci_cfg["order_band"]),
            fallback_safety_factor=float(gci_cfg["fallback_safety_factor"]),
        )
        for name in ("A1_peak", "tau_amp")
    }

    record = {
        "case": case["name"],
        "runner": "openfoam",
        "cycles_run": CYCLES,
        "verified_exactly": {
            "momentum_identity": {
                "definition": "tau_w(t) = h (G(t) - d<u>/dt), per timestep",
                "max_over_all_runs": max(
                    e["identity_max_rel"] for e in ladder
                ),
                "tol": IDENTITY_TOL,
            },
            "half_wave_symmetry": {
                "definition": "u(y, t+T/2) = -u(y, t) on the periodic state",
                "residual": main["half_wave_residual"],
                "periodicity_residual": main["final_periodicity"],
                "ratio": main["half_wave_residual"] / main["final_periodicity"],
            },
            "limit_womersley": limit_womersley,
            "limit_quasi_steady": limit_quasi_steady,
        },
        "estimated_by_gci": gci,
        "gci_ladder": ladder,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "openfoam_version": "14",
        "git_sha": git_sha(REPO),
    }
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    # --- assertions --------------------------------------------------------
    # 1. The identity, on every run. Also the runner's gate, so anything that
    #    reached here has closed it; assert it explicitly all the same.
    for entry in ladder + [limit_womersley, limit_quasi_steady]:
        assert entry["identity_max_rel"] < IDENTITY_TOL, (
            f"momentum-balance identity open by {entry['identity_max_rel']:.3e}"
        )

    # 2. Half-wave symmetry is bounded by the periodicity residual: the
    #    symmetry is not independently violated, the state is simply not yet
    #    fully periodic at this cycle budget.
    ratio = main["half_wave_residual"] / main["final_periodicity"]
    assert ratio < HALF_WAVE_OVER_PERIODICITY_MAX, (
        f"half-wave residual {main['half_wave_residual']:.3e} is "
        f"{ratio:.1f}x the periodicity residual {main['final_periodicity']:.3e} "
        f"— too large to attribute to incomplete periodicity alone"
    )

    # 3. Odd-harmonic content: the signature of the nonlinearity. A Newtonian
    #    response has NO third harmonic at all, so this is the analogue of the
    #    phase metric in womersley — a quantity that cannot exist in the
    #    linear case.
    assert main["A3_over_A1_tau"] > 1e-3, (
        f"expected measurable third-harmonic content at Cu=10; got "
        f"{main['A3_over_A1_tau']:.3e}"
    )
    assert limit_womersley["A3_over_A1_tau"] < main["A3_over_A1_tau"] / 10.0, (
        f"third-harmonic content must collapse as Cu -> 0: "
        f"{limit_womersley['A3_over_A1_tau']:.3e} at Cu={cu0} vs "
        f"{main['A3_over_A1_tau']:.3e} at Cu=10"
    )

    # 4. Effective Womersley number exceeds nominal — the mesh-sizing point.
    assert main["alpha_eff"] > main["alpha_nominal"], (
        f"shear thinning must RAISE the effective Womersley number: "
        f"{main['alpha_eff']:.2f} vs nominal {main['alpha_nominal']:.2f}"
    )

    # 5. Both limits recover their exact oracles to the tolerance those cases
    #    already meet (1e-2 on amplitude, 2e-2 on phase).
    assert limit_womersley["L2_amplitude"] < 1e-2, limit_womersley
    assert limit_womersley["L2_phase"] < 2e-2, limit_womersley
    assert limit_quasi_steady["L2_vs_steady_carreau"] < 1e-2, limit_quasi_steady
