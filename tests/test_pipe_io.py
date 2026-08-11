"""Stage A, reproducible from the repo.

The three Stage A runs (results/pipe_io_stage_a.json) came from an ad-hoc
script: the committed runner fixed nx = 4 while the recorded meshes are
8x40, 16x80, 32x160. This test closes that gap by running the COARSEST level
through the runner's now-parameterised path and checking that both recorded
findings reproduce:

  1. the velocity field converges cleanly against the exact solution
     (the profile PASSES), while
  2. the momentum identity fails to close at the ~2e-3 level (the identity
     FAILS) — the pairing that is the paper's central claim, on the
     configuration production actually uses.

The identity residual is asserted in a BAND around the recorded value, not
at it: the number is a property of the discrete solution, which is stable,
but linear-solver iteration counts differ across machines and shift it in
the last digits. A residual that came out at 1e-6 (identity suddenly
closing) or 1e-1 (something newly broken) is what this test exists to catch.
"""

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))

from betaflow.analytic import pipe  # noqa: E402
from betaflow.metrics import METRICS  # noqa: E402
from betaflow.runners import run_case  # noqa: E402

import identity_check as ic  # noqa: E402

CASE_FILE = REPO / "betaflow" / "cases" / "pipe_poiseuille_io.yaml"

# Recorded at 8x40 in results/pipe_io_stage_a.json: relative_axial 2.214e-3.
RECORDED_RESIDUAL = 2.214e-3
RESIDUAL_BAND = (1.0e-3, 4.0e-3)


@pytest.mark.slow
def test_stage_a_reproduces_from_the_repo():
    case = yaml.safe_load(CASE_FILE.read_text())
    a = float(case["geometry"]["radius"])

    r = run_case(case, runner="openfoam", n_cells=40, n_streamwise=8,
                 sampling="cell", workdir=REPO / "_runs")

    # 1. The velocity profile PASSES — this is the half of the pairing that
    #    makes the identity's failure a finding rather than a broken run.
    l2 = METRICS["L2_velocity"](
        np.asarray(r["u"]) / r["u_ref"],
        pipe.poiseuille_profile(np.asarray(r["y"]) / a),
    )
    # 2.442e-3 measured at this level when the test was written; the Stage A
    # record kept only the ORDERS (2.04, 3.02), not per-level L2, so this
    # run's own measurement is the reference. The bound is a regression
    # guard, not a physics tolerance.
    assert l2 < 3.5e-3, f"velocity L2 {l2:.3e} — the profile leg should pass"

    # 2. The momentum identity FAILS at the recorded scale, measured by the
    #    same read-only checker the production audit uses.
    casedir = REPO / "_runs" / "pipe_poiseuille_io_openfoam_n40_n_cells40_nx8_cell"
    assert casedir.exists(), f"expected case dir {casedir}"
    patches = ic.mesh_boundary(casedir)
    times = sorted(
        (p for p in casedir.iterdir()
         if p.is_dir() and p.name.replace(".", "", 1).isdigit()),
        key=lambda p: float(p.name),
    )
    terms = ic.momentum_terms(casedir, times[-1], patches)
    f_p = sum(v["pressure_force"] for k, v in terms.items()
              if k != "_missing" and "pressure_force" in v)
    f_v = sum(v["viscous_force"] for k, v in terms.items()
              if k != "_missing" and "viscous_force" in v)
    f_m = sum(v["momentum_flux"] for k, v in terms.items()
              if k != "_missing" and "momentum_flux" in v)
    residual = f_p + f_v - f_m
    scale = abs(f_p[0]) + abs(f_v[0]) + abs(f_m[0])
    relative = abs(residual[0]) / scale

    assert RESIDUAL_BAND[0] < relative < RESIDUAL_BAND[1], (
        f"momentum residual {relative:.3e} outside the recorded band "
        f"{RESIDUAL_BAND} (recorded {RECORDED_RESIDUAL:.3e}). Below the band "
        f"means the identity now CLOSES here — which would obsolete Stage A's "
        f"conclusion and needs investigating, not celebrating."
    )
    # ... and it is the same defect, not merely the same size: the closed
    # surface stays at round-off, so geometry is not the cause.
    net, area = ic.closed_surface_residual(patches)
    assert float(np.linalg.norm(net)) / area < 1e-14
