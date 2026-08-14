"""Circular-pipe cases on an axisymmetric wedge.

WHAT THIS FILE TESTS, IN ONE PARAGRAPH. A round pipe is simulated as a thin
2.5-degree slice (a "wedge" — valid because nothing varies around the axis),
and three flows are run through it: steady Poiseuille on three meshes,
steady Casson (a yield-stress blood model), and pulsatile Womersley flow.
Every measured number is written to results/pipe.json BEFORE any demand is
made, so a failure still leaves the evidence on disk. Then six separate
tests each demand one thing, so a failure names exactly what broke.

WHY THE FILE EXISTS — two purposes. First, everything before it used flat
channels, so channel assumptions could hide inside the framework's own
machinery; pushing a genuinely different geometry through tests the
framework, beyond the solver (seven quietly channel-specific assumptions
were found this way — README, "Pipe geometry"). Second, the particle and
comms cases need pipe geometry, because their published exact results are
all cylindrical.

THE CENTRAL TRAP. In a pipe the force balance is tau(r) = G r / 2; in a
channel it is tau(y) = G y. That factor of two propagates into every
profile, so the conservation identity here is written against G a / 2 —
aimed at the most likely silent geometry error. Its history of catches and
the exact solutions' citations live in betaflow/analytic/pipe.py and the
case YAMLs.
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest
import yaml

from betaflow.analytic import pipe, womersley as channel_womersley
from betaflow.metrics import METRICS
from betaflow.provenance import git_sha
from betaflow.runners import run_case

REPO = Path(__file__).resolve().parents[1]
CASES = REPO / "betaflow" / "cases"
RESULTS_FILE = REPO / "results" / "pipe.json"

# The steady case runs at each of these; every level halves the cell size.
MESH_LEVELS = (40, 80, 160)
# Acceptable band for the OBSERVED order of accuracy. "Error quarters when
# cells halve" (order 2) is exact only in the fine-mesh limit; on real
# meshes the measured order wobbles around 2, and this band accepts the
# wobble while still catching order 1 (a scheme bug) or order 0 (a floor).
P_BAND = (1.8, 2.2)
# Ceiling for conservation identities: comfortably above computer round-off
# (the measured values sit near 1e-12) and a million times below any real
# modelling error, so a genuine defect cannot hide under it.
IDENTITY_TOL = 1e-9


def _case(name):
    return yaml.safe_load((CASES / f"{name}.yaml").read_text())


# --------------------------------------------------------------------------
# One fixture runs all five solver cases ONCE and writes the record; the
# tests below only read it. (A pytest "fixture" is shared setup: the
# scope="module" flag means this function executes a single time for the
# whole file, so the expensive OpenFOAM runs are never repeated per test.)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def pipe_record():
    record = {"runner": "openfoam", "topology": "wedge"}

    # --- 1. Steady Hagen-Poiseuille at three meshes -----------------------
    # Per mesh: the profile error against the exact parabola, the wall
    # shear against G a / 2, and the Reynolds number the run realised.
    case = _case("pipe_poiseuille_steady")
    a = float(case["geometry"]["radius"])
    runs = []
    for n in MESH_LEVELS:
        r = run_case(case, runner="openfoam", n_cells=n, sampling="cell",
                     workdir=REPO / "_runs")
        m = r["meta"]
        g = m["pressure_gradient"]
        runs.append({
            "n_cells": n,
            "L2_velocity": METRICS["L2_velocity"](
                np.asarray(r["u"]) / r["u_ref"],
                pipe.poiseuille_profile(np.asarray(r["y"]) / a),
            ),
            # tau_w = G a / 2, NOT the channel's G h.
            "wss_relative": METRICS["wss_relative"](r["tau_w"], pipe.tau_wall(g, a)),
            "Re_from_meta": pipe.reynolds(m["u_mean"], a, m["nu"]),
        })
    # Observed order: log2(coarse error / fine error) per mesh pair. A
    # second-order method quarters the error when cells halve, log2(4) = 2.
    errs = [x["L2_velocity"] for x in runs]
    p_obs = [math.log2(c / f) for c, f in zip(errs, errs[1:])]
    record["poiseuille"] = {"runs": runs, "p_observed": p_obs}

    # --- 2. Steady Casson: the plug --------------------------------------
    # A Casson fluid refuses to shear below the yield stress tau_y, so an
    # inner core of radius r_p = 2 tau_y / G moves as a solid plug. The
    # realised plug ratio is recomputed from the SOLVER'S OWN discrete G.
    case_c = _case("pipe_casson_steady")
    rc = run_case(case_c, runner="openfoam", n_cells=80, sampling="cell",
                  workdir=REPO / "_runs")
    mc = rc["meta"]
    gc = mc["pressure_gradient"]
    xi_c_measured = pipe.plug_radius_ratio(mc["tau0"], gc, a)
    record["casson"] = {
        "n_cells": mc["mesh_level"],
        "xi_c_target": float(case_c["nondim"]["xi_c"]),
        "xi_c_realised": xi_c_measured,
        "plug_radius": pipe.plug_radius(mc["tau0"], gc),
        "identity": mc["identity"],
        "L2_velocity": METRICS["L2_velocity"](
            np.asarray(rc["u"]) / rc["u_ref"],
            pipe.casson_profile(np.asarray(rc["y"]) / a, xi_c_measured),
        ),
    }

    # --- 3. Pulsatile Womersley, plus a planted impostor ------------------
    # The driving force oscillates like a heartbeat, so at each radius the
    # velocity swings with an AMPLITUDE (size of the swing) and a PHASE
    # (time lag behind the force). The exact answer in a cylinder involves
    # the Bessel function J0 — the role cosh plays in a flat channel. The
    # channel's cosh solution is ALSO evaluated on the same points, as a
    # deliberately planted impostor: the kernel mix-up happened for real
    # during development, and this keeps the trap armed.
    case_w = _case("pipe_womersley_pulsatile")
    rw = run_case(case_w, runner="openfoam", workdir=REPO / "_runs", max_cycles=10)
    mw = rw["meta"]
    r_over_a = np.asarray(rw["y"]) / a
    exact = pipe.womersley_profile(r_over_a, mw["alpha"])
    tau_exact = pipe.womersley_wall_shear(mw["alpha"])
    wrong_kernel = channel_womersley.complex_profile(r_over_a, mw["alpha"])
    record["womersley"] = {
        "n_cells": mw["mesh_level"],
        "cells_per_stokes_layer": mw["cells_per_stokes_layer"],
        "viscous_fourier": mw["viscous_fourier"],
        "identity_max_rel": mw["identity_max_rel"],
        "cycles_to_periodic": mw["cycles_to_periodic"],
        "L2_amplitude": METRICS["L2_amplitude"](
            rw["u_amp"] / rw["u_ref"], np.abs(exact)
        ),
        "L2_phase": METRICS["L2_phase"](
            rw["u_phase"], np.angle(exact), np.abs(exact) ** 2
        ),
        "wss_amp_relative": METRICS["wss_amp_relative"](
            rw["tau_amp"] / rw["tau_ref"], np.abs(tau_exact)
        ),
        "wss_phase_error": float(
            abs(np.angle(np.exp(1j * (rw["tau_phase"] - np.angle(tau_exact)))))
        ),
        "L2_amplitude_wrong_cosh_kernel": METRICS["L2_amplitude"](
            rw["u_amp"] / rw["u_ref"], np.abs(wrong_kernel)
        ),
    }

    # --- 4. Kernel discrimination, no solver ------------------------------
    # Pure mathematics: how far apart are the pipe (J0) and channel (cosh)
    # solutions, per metric, across the physiological alpha range? This is
    # the blind-spot study rebuilt inside the test so its numbers can never
    # drift from the claim they support.
    r_fine = np.linspace(0.0, 1.0, 201)
    discrimination = []
    for al in (2.0, 5.0, 10.0, 20.0):
        j0 = pipe.womersley_profile(r_fine, al)
        co = channel_womersley.complex_profile(r_fine, al)
        discrimination.append({
            "alpha": al,
            "amplitude_misfit": METRICS["L2_amplitude"](np.abs(co), np.abs(j0)),
            "phase_misfit": METRICS["L2_phase"](
                np.angle(co), np.angle(j0), np.abs(j0) ** 2
            ),
            "wss_amplitude_misfit": float(
                abs(abs(channel_womersley.complex_wall_shear(al))
                    - abs(pipe.womersley_wall_shear(al)))
                / abs(pipe.womersley_wall_shear(al))
            ),
        })
    record["kernel_discrimination"] = discrimination

    # Evidence on disk BEFORE any demand is made.
    record["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["openfoam_version"] = "14"
    record["git_sha"] = git_sha(REPO)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")
    return record


# --------------------------------------------------------------------------
# The demands. Each test checks one thing and says why it matters.
# --------------------------------------------------------------------------

def test_steady_wall_shear_identity(pipe_record):
    """tau_w must equal G a / 2 at round-off, at EVERY mesh.

    The steady periodic pipe leaves the solver no freedom: the push
    (G x volume) must balance wall friction exactly, which pins tau_w
    independent of resolution. This exact check caught the wedge-faceting
    bias (1 - cos(2.5 deg) = 9.5e-4, refinement-independent — the flat
    chord standing in for the curved wall) and a transient lever-arm error
    of exactly 100% (the channel's h carried in for the pipe's a/2).
    """
    for x in pipe_record["poiseuille"]["runs"]:
        assert x["wss_relative"] < IDENTITY_TOL, (
            f"tau_w != G a/2 at N={x['n_cells']}: {x['wss_relative']:.3e}"
        )


def test_steady_reynolds_definition(pipe_record):
    """Runner and exact solution must MEAN the same thing by Re = 100.

    The pipe convention uses the diameter 2a; the channel uses the full gap
    2h. Using the wrong one silently changes Re by a factor of 2, so the
    definition agreement is pinned to twelve digits before any physics is
    compared.
    """
    case = _case("pipe_poiseuille_steady")
    for x in pipe_record["poiseuille"]["runs"]:
        np.testing.assert_allclose(
            x["Re_from_meta"], case["nondim"]["Re"], rtol=1e-12,
            err_msg="runner and analytic reference disagree on the pipe Re definition (2a, not 2h)",
        )


def test_steady_convergence_order(pipe_record):
    """The profile error must shrink at the PREDICTED rate (order ~2).

    A small error at one mesh proves little; quartering when cells halve is
    the second-order signature, and matches the channel result.
    """
    pairs = zip(MESH_LEVELS, MESH_LEVELS[1:])
    for pair, p_val in zip(pairs, pipe_record["poiseuille"]["p_observed"]):
        assert P_BAND[0] < p_val < P_BAND[1], (
            f"pipe Poiseuille order p={p_val:.3f} outside {P_BAND} for {pair}"
        )


def test_casson_plug_and_profile(pipe_record):
    """The realised plug must match the request; the profile must fit.

    First a configuration check: the plug ratio recomputed from the
    solver's own discrete G must hit the requested target within 0.1%
    (did the run set up the physics it claimed?). Then the accuracy
    question: the velocity shape, plug included, within the case tolerance.
    The solver's own conservation identity must also hold.
    """
    case_c = _case("pipe_casson_steady")
    c = pipe_record["casson"]
    assert c["identity"] < IDENTITY_TOL
    assert abs(c["xi_c_realised"] / c["xi_c_target"] - 1.0) < 1e-3
    assert c["L2_velocity"] < float(case_c["metrics"][0]["tol"])


def test_womersley_fits_the_pipe_kernel(pipe_record):
    """The J0 solution must fit within the case tolerances; identity holds."""
    case_w = _case("pipe_womersley_pulsatile")
    w = pipe_record["womersley"]
    assert w["identity_max_rel"] < 1e-6
    assert w["L2_amplitude"] < float(case_w["metrics"][0]["tol"])
    assert w["L2_phase"] < float(case_w["metrics"][1]["tol"])
    assert w["wss_amp_relative"] < float(case_w["metrics"][2]["tol"])


def test_kernel_swap_is_caught_by_wall_shear_only(pipe_record):
    """The geometry distinction must be visible — in the right metric.

    At alpha = 10 the channel kernel misfits the pipe PROFILE by 9.1e-3 in
    amplitude and 1.5e-2 in phase — BOTH inside this case's own tolerances
    (1e-2, 2e-2) — so a profile-only validation would pass a swapped
    kernel. Wall shear misfits by 48%, because the two transfer functions
    differ by a factor approaching exactly 2 at large alpha.

    Two matched demands follow. Wall shear must separate the kernels
    DECISIVELY (else the geometry distinction is untested). And the profile
    misfit must stay BELOW tolerance — that asserts the blind spot itself,
    so if solver tolerances or the mathematics ever change enough that
    profiles become discriminating, this fails and says the recorded claim
    has gone stale.
    """
    case_w = _case("pipe_womersley_pulsatile")
    at_alpha = next(d for d in pipe_record["kernel_discrimination"]
                    if d["alpha"] == 10.0)
    wss_tol = float(case_w["metrics"][2]["tol"])
    assert at_alpha["wss_amplitude_misfit"] > 10.0 * wss_tol, (
        f"wall shear should separate the two kernels decisively; misfit "
        f"{at_alpha['wss_amplitude_misfit']:.3e} vs tol {wss_tol:.0e}"
    )
    assert at_alpha["amplitude_misfit"] < float(case_w["metrics"][0]["tol"]), (
        "recorded as a KNOWN BLIND SPOT: if the amplitude misfit between "
        "kernels ever exceeds the case tolerance, this note is stale and the "
        "profile metric has become discriminating after all"
    )
