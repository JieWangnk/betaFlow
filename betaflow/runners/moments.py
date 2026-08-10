"""Aris moment hierarchy on the cross-section. No solver, no axial mesh.

THE THIRD NON-OPENFOAM RUNNER, and the reference leg for Eulerian scalar
transport. `langevin.py` answers the particle oracles without a CFD solver;
this module answers the CONCENTRATION oracles the same way.

WHAT IT SOLVES, and why that is the interesting part. Multiplying the
advection-diffusion equation by x^p and integrating over the whole axial line
kills the axial coordinate exactly: the advective term integrates by parts to
p u c_{p-1}, the axial diffusive term to p(p-1) D c_{p-2}, and the boundary
terms vanish because the pulse decays. What is left is a hierarchy of purely
TRANSVERSE problems,

    d c_p / dt = D lap_perp c_p + p u c_{p-1} + p(p-1) D c_{p-2},

with no-flux walls, where c_p(y_perp, t) = integral x^p c dx. The axial
moments then follow as m_p = <c_p>. See Aris (1956).

    THERE IS NO AXIAL DISCRETISATION ANYWHERE IN THIS RUNNER.

That is a scoping statement, not a boast, and it cuts both ways.

  It CAN test the transverse operator to high precision, and the transverse
  operator is exactly what D_eff, the variance intercept and the third
  cumulant all depend on. Every anchor in `analytic/advection_diffusion.py`
  except the free-space Green's function is measurable here at a cost of a few
  hundred cells and one matrix exponential.

  It CANNOT test a solver's axial transport at all: not its advection scheme,
  not its numerical diffusion, not its boundedness. A code could be hopeless
  in x and this runner would never know, because it does not represent x.
  Comparing an OpenFOAM or lattice-Boltzmann scalar run against these numbers
  therefore isolates the transverse physics; the axial behaviour needs the
  Green's function anchor and a real axial mesh.

TIME INTEGRATION IS EXACT. The hierarchy is linear with constant coefficients,
so the discrete system is advanced by a matrix exponential of the assembled
block operator. There is no timestep error to confound with the spatial error,
which is what lets the variance INTERCEPT — a subtle O(L^4 U^2/D^2) constant
that a fitted D_eff would otherwise absorb — be measured to 1e-5 at 200 cells
and shown to converge at second order in the transverse mesh.

THE TRANSVERSE OPERATOR IS ASSEMBLED FROM FACE FLUXES, so it is conservative
by construction: interior faces cancel pairwise and the no-flux wall
contributes nothing, giving 1^T (V L) = 0 identically. m_0 is then constant to
round-off for any mesh and any timestep, which is the discrete statement of
anchor 1 and the first thing `run` reports.

CITATIONS
  Aris, R. (1956), Proc. R. Soc. A 235(1200):67-77 — the moment hierarchy.
  All exact answers come from betaflow/analytic/advection_diffusion.py, which
  carries its own citations and its own self-verification.
"""

import numpy as np
from scipy.linalg import expm

from betaflow.analytic import advection_diffusion as ad

# Moments carried. p = 3 is needed for the third cumulant; going higher costs
# a block per moment and buys an anchor nobody has an exact answer for.
N_MOMENTS = 4


def _transverse_operator(geometry, n_cells, diffusivity, length):
    """Conservative finite-volume no-flux Laplacian on the cross-section.

    Returns (L, cell_centres, volumes). Built face by face, so summing the
    rows against the volumes gives exactly zero: mass cannot be created by
    this operator regardless of mesh or geometry.
    """
    if geometry == "pipe":
        edges = np.linspace(0.0, length, n_cells + 1)
        volume = np.pi * (edges[1:] ** 2 - edges[:-1] ** 2)
        face_area = 2.0 * np.pi * edges
    else:
        edges = np.linspace(-length, length, n_cells + 1)
        volume = np.diff(edges)
        face_area = np.ones(n_cells + 1)
    centre = 0.5 * (edges[1:] + edges[:-1])

    op = np.zeros((n_cells, n_cells))
    spacing = np.diff(centre)
    for i in range(n_cells - 1):
        # Interior faces only. The wall face is never visited, which IS the
        # no-flux condition — imposing it afterwards would be a second chance
        # to get it wrong.
        g = diffusivity * face_area[i + 1] / spacing[i]
        op[i, i] -= g
        op[i, i + 1] += g
        op[i + 1, i] += g
        op[i + 1, i + 1] -= g
    return op / volume[:, None], centre, volume


def _initial_condition(geometry, centre, length, volume, release, xi_release):
    """c_0 at t = 0, normalised to unit released mass.

    'plane'  — cross-sectionally uniform, the Eulerian twin of the area-uniform
               seeding taylor_aris uses. Centroid is then U t at ALL times.
    'point'  — concentrated in the cell containing xi_release. The centroid
               starts at the LOCAL velocity and relaxes, leaving the permanent
               offset `analytic.advection_diffusion.centroid_offset` predicts.
    """
    c0 = np.zeros(len(centre))
    if release == "plane":
        c0[:] = 1.0
        xi_actual = float("nan")
    elif release == "point":
        # A "point" release occupies one cell, whose centre is not exactly the
        # requested xi. The cell function varies across that offset, so the
        # ACTUAL centre is returned and the exact answer is evaluated there.
        # Comparing against b(xi_requested) instead would charge a
        # discretisation artefact to the physics -- it costs 4.7% at N=200 on
        # the channel, which is large enough to look like a real discrepancy.
        idx = int(np.argmin(np.abs(centre / length - xi_release)))
        c0[idx] = 1.0 / volume[idx]
        xi_actual = float(centre[idx] / length)
    else:
        raise ValueError(f"release must be 'plane' or 'point', got {release!r}")
    return c0 / float(np.sum(c0 * volume) / np.sum(volume)), xi_actual


def run(
    case,
    n_cells=200,
    n_steps=120,
    t_end_over_tau=6.0,
    release="plane",
    xi_release=0.0,
    **params,
):
    """Advance the moment hierarchy and report the axial moments.

    Returns t, the raw moments, the central variance and third cumulant, and
    every exact answer the oracle supplies for this configuration, so a metric
    never has to re-derive one.
    """
    phys = case["physical"]
    geometry = case["geometry"]["type"]
    length = float(case["geometry"]["length_scale"])
    diffusivity = float(phys["diffusivity"])
    u_mean = float(phys["mean_velocity"])

    op, centre, volume = _transverse_operator(geometry, n_cells, diffusivity, length)
    weight = volume / volume.sum()
    u = u_mean * (1.0 + ad.velocity_deviation(centre / length, geometry))

    # THE DISCRETE MEAN VELOCITY, which is not exactly u_mean. Sampling a
    # quadratic profile at cell centres and volume-averaging leaves an
    # O(dxi^2) error, so the discrete centroid advances at u_disc rather than
    # at the nominal U. Comparing a centroid against U t therefore accumulates
    # a LINEAR DRIFT that looks like an offset error when read at one late
    # time -- it doubles when the run doubles in length. Reporting u_disc and
    # comparing against it removes the artefact rather than tolerating it,
    # which is the same move the wedge cases make with G_disc.
    u_discrete = float(weight @ u)

    n = n_cells
    block = np.zeros((N_MOMENTS * n, N_MOMENTS * n))
    for p in range(N_MOMENTS):
        block[p * n : (p + 1) * n, p * n : (p + 1) * n] = op
        if p >= 1:
            block[p * n : (p + 1) * n, (p - 1) * n : p * n] += p * np.diag(u)
        if p >= 2:
            block[p * n : (p + 1) * n, (p - 2) * n : (p - 1) * n] += (
                p * (p - 1) * diffusivity * np.eye(n)
            )

    tau_r = ad.transverse_relaxation_time(length, diffusivity)
    dt = t_end_over_tau * tau_r / n_steps
    propagator = expm(block * dt)

    state = np.zeros(N_MOMENTS * n)
    state[:n], xi_actual = _initial_condition(
        geometry, centre, length, volume, release, xi_release
    )

    times = np.zeros(n_steps + 1)
    raw = np.zeros((N_MOMENTS, n_steps + 1))
    for k in range(n_steps + 1):
        if k:
            state = propagator @ state
            times[k] = k * dt
        for p in range(N_MOMENTS):
            raw[p, k] = float(weight @ state[p * n : (p + 1) * n])

    m0 = raw[0]
    mu = raw / m0  # moments about the origin, per unit mass
    variance = mu[2] - mu[1] ** 2
    third_cumulant = mu[3] - 3.0 * mu[1] * mu[2] + 2.0 * mu[1] ** 3

    return {
        "t": times,
        "m0": m0,
        "centroid": mu[1],
        "var_x": variance,
        "third_cumulant": third_cumulant,
        # Exact answers, from the oracle, for whatever this case configured.
        "d_eff_expected": ad.d_eff(diffusivity, length, u_mean, geometry),
        "intercept_expected": ad.variance_intercept(
            length, u_mean, diffusivity, geometry
        ),
        "kappa3_slope_expected": ad.SKEWNESS_FACTOR[geometry]
        * length**4
        * u_mean**3
        / diffusivity**2,
        "u_mean_discrete": u_discrete,
        "centroid_offset_expected": (
            0.0
            if release == "plane"
            else ad.centroid_offset(xi_actual, length, u_mean, diffusivity, geometry)
        ),
        "meta": {
            "solver": "moments",
            "note": "Aris moment hierarchy; NO axial discretisation exists here",
            "geometry": geometry,
            "n_cells": n_cells,
            "n_steps": n_steps,
            "dt": dt,
            "release": release,
            "xi_release": xi_release,
            "xi_actual": xi_actual,
            "tau_r": tau_r,
            "t_end_over_tau": t_end_over_tau,
            "peclet": ad.peclet(u_mean, length, diffusivity),
            "balance_peclet": ad.balance_peclet(geometry),
            "diffusivity": diffusivity,
            "mean_velocity": u_mean,
            "u_mean_discrete": u_discrete,
            "u_mean_discretisation_error": abs(u_discrete / u_mean - 1.0),
            "length_scale": length,
            # Conservation is a property of the operator, not of convergence,
            # so it is reported rather than asserted here.
            "mass_drift": float(np.max(np.abs(m0 / m0[0] - 1.0))),
            "operator_row_sum": float(
                np.max(np.abs(volume @ op))
                / (diffusivity * volume.sum() / length**2)
            ),
        },
    }
