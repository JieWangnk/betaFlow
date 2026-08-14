"""Circular-pipe cases on an axisymmetric wedge.

Purpose is twofold: test whether the harness abstractions hold for any
geometry or are quietly channel-specific, and open the route to particle
transport, whose published analytic references are cylindrical.

Every analytic reference used here is cited in betaflow/analytic/pipe.py and in the case
YAML. The force balance tau(r) = G r / 2 — a factor of two from the channel's
tau(y) = G y — propagates into every profile, and the conservation identity is
written against G a / 2 precisely because that factor is the most likely
silent geometry error.
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

MESH_LEVELS = (40, 80, 160)
P_BAND = (1.8, 2.2)
IDENTITY_TOL = 1e-9


def _case(name):
    return yaml.safe_load((CASES / f"{name}.yaml").read_text())


def test_pipe_geometry():
    record = {"runner": "openfoam", "topology": "wedge"}

    # --- 1. Steady Hagen-Poiseuille: order of accuracy + identity ---------
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
    errs = [x["L2_velocity"] for x in runs]
    p_obs = [math.log2(c / f) for c, f in zip(errs, errs[1:])]
    record["poiseuille"] = {"runs": runs, "p_observed": p_obs}

    # --- 2. Steady Casson: plug radius r_p = 2 tau_y / G ------------------
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

    # --- 3. Pulsatile: the J0 kernel, in the geometry where it is correct -
    case_w = _case("pipe_womersley_pulsatile")
    rw = run_case(case_w, runner="openfoam", workdir=REPO / "_runs", max_cycles=10)
    mw = rw["meta"]
    r_over_a = np.asarray(rw["y"]) / a
    exact = pipe.womersley_profile(r_over_a, mw["alpha"])
    tau_exact = pipe.womersley_wall_shear(mw["alpha"])
    # A channel cosh kernel evaluated on the same stations, as a guard: if the
    # two were interchangeable this would fit too, and the earlier
    # channel/pipe kernel confusion would be undetectable.
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

    # How well can each metric even TELL the two kernels apart? Analytic reference-to-
    # analytic reference, so no solver runs are needed. The answer is uncomfortable: at
    # the case's alpha the profile metrics cannot, but wall shear can.
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

    record["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    record["openfoam_version"] = "14"
    record["git_sha"] = git_sha(REPO)
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    RESULTS_FILE.write_text(json.dumps(record, indent=2) + "\n")

    # --- assertions -------------------------------------------------------
    # The conservation identity, now against G a / 2. It is what caught the
    # wedge faceting bias (1 - cos(theta) = 9.5e-4, refinement-independent)
    # and the transient lever-arm error (exactly 1.0, a 100% error).
    for x in runs:
        assert x["wss_relative"] < IDENTITY_TOL, (
            f"tau_w != G a/2 at N={x['n_cells']}: {x['wss_relative']:.3e}"
        )
        np.testing.assert_allclose(
            x["Re_from_meta"], case["nondim"]["Re"], rtol=1e-12,
            err_msg="runner and analytic reference disagree on the pipe Re definition (2a, not 2h)",
        )
    assert record["casson"]["identity"] < IDENTITY_TOL
    assert record["womersley"]["identity_max_rel"] < 1e-6

    # Second-order convergence, matching the channel result.
    for pair, p_val in zip(zip(MESH_LEVELS, MESH_LEVELS[1:]), p_obs):
        assert P_BAND[0] < p_val < P_BAND[1], (
            f"pipe Poiseuille order p={p_val:.3f} outside {P_BAND} for {pair}"
        )

    # Casson: the realised plug ratio must be the requested one, via
    # r_p = 2 tau_y / G computed from the DISCRETE G.
    assert abs(xi_c_measured / record["casson"]["xi_c_target"] - 1.0) < 1e-3
    assert record["casson"]["L2_velocity"] < float(case_c["metrics"][0]["tol"])

    # Womersley: J0 fits, cosh does not. The wrong kernel must be worse by
    # orders of magnitude, or the geometry distinction is not being tested.
    w = record["womersley"]
    assert w["L2_amplitude"] < float(case_w["metrics"][0]["tol"])
    assert w["L2_phase"] < float(case_w["metrics"][1]["tol"])
    assert w["wss_amp_relative"] < float(case_w["metrics"][2]["tol"])
    # Which metric actually catches a channel/pipe kernel swap? At alpha = 10
    # the cosh kernel misfits the pipe profile by 9.1e-3 in amplitude and
    # 1.5e-2 in phase — BOTH BELOW this case's own tolerances (1e-2, 2e-2) —
    # so a profile-only validation would pass a swapped kernel. Wall shear
    # misfits by 48%, because tanh(K)/K and 2 J1(b)/(b J0(b)) differ by a
    # factor approaching exactly 2 at large alpha. Assert on the metric that
    # can actually tell them apart.
    at_alpha = next(d for d in discrimination if d["alpha"] == 10.0)
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
