"""Null test for tools/identity_check.py — the CHECKER, not the physics.

An instrument needs its own calibration. The mass-tier parser reported a
spurious 33% imbalance because it ran past a patch block and double-counted
the inlet as the wall; that was caught only because 33% is implausible. The
same bug producing 1e-4 would have read as mild under-convergence and been
accepted. So the checker is run against a case whose answer is known exactly
before it is pointed at anything else — the same argument as building the
analytic reference before the solver, applied one level up.

The fixture is a committed 40-cell channel: no solver needed, pure file
parsing, milliseconds. That is why this sits in the analytic reference CI tier.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.identity_check import (  # noqa: E402
    check_case,
    closed_surface_residual,
    domain_volume,
    mesh_boundary,
    momentum_closure,
    read_mean_velocity_force,
)

pytestmark = pytest.mark.analytic

FIXTURE = REPO / "tests" / "fixtures" / "poiseuille_n40"

# Known exactly for this fixture: a 2h x L x w box with h=1, L=0.5, w=0.1.
EXPECTED_VOLUME = 0.1
EXPECTED_AREA = 2.5


def test_closed_surface_is_the_first_check():
    """sum of Sf over every boundary face vanishes — pure mesh, no fields.

    This is the check that would have caught the mass-tier parser bug
    immediately: a missing or duplicated patch shows up here and nowhere else.
    """
    patches = mesh_boundary(FIXTURE)
    net, area = closed_surface_residual(patches)
    assert area == pytest.approx(EXPECTED_AREA, rel=1e-12)
    assert np.linalg.norm(net) / area < 1e-14, (
        f"closed-surface residual {net} is not round-off — a patch is missing "
        f"or double-counted"
    )


def test_volume_from_boundary_only():
    """V = (1/3) closed-integral x.n dS, so no cell data is needed."""
    assert domain_volume(mesh_boundary(FIXTURE)) == pytest.approx(
        EXPECTED_VOLUME, rel=1e-12
    )


def test_mass_closure():
    """Flux balance on a case whose inlet/outlet are cyclic: identically zero."""
    report = check_case(FIXTURE)
    assert report["times"], "fixture has no written phi"
    last = report["times"][-1]
    # Cyclic pairs and a zero-flux wall: every patch contributes zero, so the
    # imbalance is exactly zero rather than merely small.
    assert abs(last["net_flux"]) < 1e-18


def test_momentum_closure_null():
    """The momentum identity on a case with an exactly known answer.

    A force-driven periodic channel balances wall traction against the volume
    source: the identity collapses to tau_w = G h, i.e. F_wall + G V = 0.
    Reported PER COMPONENT — a single component failing localises the error in
    a way a norm does not.
    """
    g = read_mean_velocity_force(FIXTURE)
    assert g is not None, "fixture log must record the applied pressure gradient"
    result = momentum_closure(FIXTURE, body_force=[g, 0.0, 0.0])

    # The instrument must reproduce the case's own identity result (~1e-12 or
    # better); anything larger is the instrument, not the data.
    residual = np.array(result["residual_relative_per_component"])
    assert residual.max() < 1e-12, (
        f"per-component residual {residual} exceeds the level this case "
        f"already achieves — fix the instrument before trusting it on data"
    )
    # And the physics it encodes: wall traction equals the body force.
    viscous = np.array(result["terms"]["viscous_force"])
    source = np.array(result["body_force_term"])
    assert viscous[0] == pytest.approx(-source[0], rel=1e-9)
    assert not result["missing_boundary_values"]["p"], (
        "a patch had no reconstructable pressure value; omitting one makes "
        "the traction integral reference-dependent"
    )
