"""A minimal lattice-Boltzmann advection-diffusion solver. Pure numpy.

THE FIFTH RUNNER, and the measurement leg of the `lattice_boltzmann` analytic reference.
That module states what an LBM ADE code must satisfy — D = c_s^2 (tau - 1/2),
the Ma^2 depletion law, the anti-bounce-back Dirichlet slip and its magic
parameter — and until this runner existed those statements were checked only
by algebra (symbolic dispersion relations, 50-digit eigenvalue expansions).
This module is an ACTUAL collide-and-stream lattice, so the analytic reference's claims
are now confronted with the thing they describe, in-repo, with no external
dependency. When OpenLB enters, its output is compared against the same
analytic reference through the same metrics; this runner is the reference implementation
that establishes the analytic reference and the harness agree before a third party is
measured against them.

TWO EXPERIMENTS, selected by the case's `experiment` key.

  "dispersion"     A Gaussian pulse on a periodic ring, advected and
                   diffused; the measured variance growth rate IS the
                   scheme's effective diffusivity. This is the configuration
                   in which D = c_s^2 (tau - 1/2) and the Ma^2 law
                   D_eff = (c_s^2 - u^2)(tau - 1/2) were lattice-verified to
                   3.7e-9 during analytic reference construction.

  "dirichlet_slip" Steady pure diffusion with a uniform source between two
                   anti-bounce-back Dirichlet walls. The analytic solution is
                   a parabola; the LBM solution is the parabola plus a
                   spurious UNIFORM offset phi_s that vanishes only at
                   Ginzburg's magic parameter Lambda = (tau - 1/2)^2 = 3/16
                   (arXiv:1603.09577 Eqs. 71-73, re-verified symbolically in
                   the analytic reference). This experiment MEASURES that offset: its
                   sign flip across tau = 1/2 + sqrt(3)/4, its 1/N^2 decay
                   at fixed Delta_phi, and its uniformity across the domain.

SOURCE-SCHEME FINDINGS, measured during construction (the caveat that stood
here anticipated scheme dependence; the measurement then resolved it):

  * With the SECOND-ORDER source scheme — amplitude w_i S (1 - 1/(2 tau))
    and the scalar redefined as phi = sum_i g_i + S/2 — the slip vanishes at
    tau = 1/2 + sqrt(3)/4 EXACTLY (|slip| < 2e-13 at the default steady
    tolerance — the iteration floor), and equals the published formula at
    every tau ONCE THE
    CONVENTION IS MATCHED: the paper's N is the HALF-width in lattice
    spacings, so slip = (Delta_phi / (12 (N/2)^2)) * 16 * [Lambda - 3/16].
    The first comparison was 4x off at every tau — a constant ratio, which
    is the signature of a length-convention mismatch, not physics.

  * The SIMPLE source scheme (amplitude w_i S, phi = sum g) shifts the whole
    slip curve by EXACTLY -S/2, at every tau and every N: it is the missing
    S/2 of the scalar redefinition and nothing else. Its apparent
    zero-crossing at tau = 1.077 is therefore an artefact of comparing an
    uncorrected scalar against the exact solution.

  * At FIXED LATTICE SOURCE S the slip is N-INDEPENDENT, and that is the
    published 1/N^2 law, not a contradiction of it: Delta_phi = S N^2/(8 D)
    grows as N^2 and cancels the 1/N^2. The law is 1/N^2 at fixed
    Delta_phi — refining the mesh on a fixed physical problem — and the
    first N-sweep here held the wrong thing fixed.

STEP BOOKKEEPING. `_variance_slope` counts COMPLETED lattice updates. The
first in-repo measurement of the dispersion experiment divided by steps//2
when 751 updates had completed, and the resulting constant 749/750 deficit
was mis-attributed to a plausible k^4-truncation story until a pulse-width
sweep refuted it. The elapsed count is now taken from the loop variable at
the moment of measurement, never recomputed.
"""

import numpy as np

from betaflow.analytic import lattice_boltzmann as lb

# Velocity components along the transport axis, per supported lattice. D2Q5's
# two transverse populations carry no x-momentum; with a y-uniform state they
# relax without affecting x-transport, so the ring problem stays 1-D.
_AXIS_VELOCITIES = {
    "D1Q3": [0, 1, -1],
    "D2Q5": [0, 1, -1, 0, 0],
}


def _weights(lattice, omega):
    """Weights for the named lattice; reduced sets come from the analytic reference's
    omega-families so runner and analytic reference cannot disagree on the convention."""
    if lattice == "D1Q3":
        if omega is not None:
            raise ValueError("D1Q3 has no free rest weight; omit omega")
        return np.array([2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0])
    if lattice == "D2Q5":
        return np.array(lb.d2q5_weights(2.0 / 3.0 if omega is None else omega))
    raise ValueError(f"lattice must be one of {tuple(_AXIS_VELOCITIES)}, got {lattice!r}")


def _cs2(lattice, weights):
    """c_s^2 from the weights ACTUALLY IN USE — the analytic reference's own rule that a
    name-keyed constant is the trap (OpenLB's D2Q5 ships with 1/5)."""
    e = _AXIS_VELOCITIES[lattice]
    if lattice == "D1Q3":
        return float(sum(w * ei**2 for w, ei in zip(weights, e)))
    # D2Q5: c_s^2 = sum_i w_i |e_i|^2 / d. All four moving populations have
    # |e|^2 = 1, so the sum is (1 - w_0), and d = 2: c_s^2 = omega / 2.
    return float(1.0 - weights[0]) / 2.0


def _equilibrium(rho, u, cs2, weights, velocities, order):
    out = np.empty((len(weights), len(rho)))
    for i, (wi, ei) in enumerate(zip(weights, velocities)):
        c = 1.0 + ei * u / cs2
        if order == 2:
            c += (ei**2 - cs2) * u**2 / (2.0 * cs2**2)
        out[i] = wi * rho * c
    return out


def _circular_variance(rho, n):
    x = np.arange(n, dtype=float)
    theta = 2.0 * np.pi * x / n
    mean = np.angle((rho * np.exp(1j * theta)).sum()) / (2.0 * np.pi) * n % n
    d = (x - mean + n / 2.0) % n - n / 2.0
    return float((rho * d**2).sum() / rho.sum())


def _run_dispersion(lattice, tau, u, equilibrium_order, n_cells, steps, sigma0, omega):
    weights = _weights(lattice, omega)
    velocities = _AXIS_VELOCITIES[lattice]
    cs2 = _cs2(lattice, weights)

    x = np.arange(n_cells, dtype=float)
    rho = np.exp(-((x - n_cells / 2.0) ** 2) / (2.0 * sigma0**2))
    g = _equilibrium(rho, u, cs2, weights, velocities, equilibrium_order)

    mark = steps // 2
    halfway = completed_at_mark = None
    for t in range(steps):
        rho = g.sum(axis=0)
        g += -(g - _equilibrium(rho, u, cs2, weights, velocities, equilibrium_order)) / tau
        for i, ei in enumerate(velocities):
            if ei:
                g[i] = np.roll(g[i], int(ei))
        if t == mark:
            halfway = _circular_variance(g.sum(axis=0), n_cells)
            completed_at_mark = t + 1  # updates COMPLETED, not the loop index
    final = _circular_variance(g.sum(axis=0), n_cells)
    d_measured = (final - halfway) / (2.0 * (steps - completed_at_mark))

    return {
        "d_measured": d_measured,
        "cs2": cs2,
        # Both predictions, so a test never re-derives either.
        "d_expected_first_order": (cs2 - u**2) * (tau - 0.5),
        "d_expected_standard": cs2 * (tau - 0.5),
        "mass_drift": float(abs(g.sum() / rho.sum() - 1.0)),
    }


def _run_dirichlet_slip(tau, n_cells, source, phi_wall, tol, max_steps,
                        source_scheme="corrected"):
    """Steady diffusion + uniform source between anti-bounce-back walls.

    D1Q3 only: the slip formula is one-dimensional. Nodes sit at
    y_j = j + 1/2 with the walls at y = 0 and y = N — the halfway placement
    the anti-bounce-back rule assumes. Exact steady solution:
    phi = phi_wall + (S / 2D) y (N - y).

    source_scheme: "corrected" adds w_i S (1 - 1/(2 tau)) and defines
    phi = sum_i g_i + S/2 (He-Luo second-order treatment); "simple" adds
    w_i S with phi = sum_i g_i, whose slip is offset by exactly -S/2. Both
    are kept because their DIFFERENCE is itself a measured result.
    """
    weights = _weights("D1Q3", None)
    velocities = _AXIS_VELOCITIES["D1Q3"]
    cs2 = _cs2("D1Q3", weights)
    diffusivity = cs2 * (tau - 0.5)
    if source_scheme == "corrected":
        source_amplitude = source * (1.0 - 1.0 / (2.0 * tau))
        phi_shift = source / 2.0
    elif source_scheme == "simple":
        source_amplitude = source
        phi_shift = 0.0
    else:
        raise ValueError(f"source_scheme must be 'corrected' or 'simple', got {source_scheme!r}")

    phi = np.full(n_cells, float(phi_wall))
    g = _equilibrium(phi, 0.0, cs2, weights, velocities, 1)

    check_every = 200
    previous = phi.copy()
    steps_run = 0
    for t in range(max_steps):
        phi = g.sum(axis=0) + phi_shift
        g += -(g - _equilibrium(phi, 0.0, cs2, weights, velocities, 1)) / tau
        g += (weights * source_amplitude)[:, None]
        # stream interior
        out_plus = g[1].copy()
        out_minus = g[2].copy()
        g[1][1:] = out_plus[:-1]
        g[2][:-1] = out_minus[1:]
        # anti-bounce-back at both walls: incoming = -outgoing + 2 w phi_wall
        g[1][0] = -out_minus[0] + 2.0 * weights[1] * phi_wall
        g[2][-1] = -out_plus[-1] + 2.0 * weights[2] * phi_wall
        steps_run = t + 1
        if steps_run % check_every == 0:
            current = g.sum(axis=0)
            scale = max(float(np.max(np.abs(current))), 1e-300)
            if float(np.max(np.abs(current - previous))) < tol * scale:
                break
            previous = current

    phi = g.sum(axis=0) + phi_shift
    y = np.arange(n_cells, dtype=float) + 0.5
    exact = phi_wall + source / (2.0 * diffusivity) * y * (n_cells - y)
    deviation = phi - exact
    offset = float(deviation.mean())
    peak = float(source * n_cells**2 / (8.0 * diffusivity))  # Delta phi

    return {
        "slip_measured": offset,
        # The published formula UNDER THE MATCHED CONVENTION: the paper's N
        # is the half-width in lattice spacings, i.e. n_cells / 2 here. The
        # unmatched form is off by exactly 4 at every tau, which is how the
        # convention was identified.
        "slip_published_form": float(
            peak
            / (12.0 * (n_cells / 2.0) ** 2)
            * 16.0
            * (lb.magic_lambda(tau) - 3.0 / 16.0)
        ),
        "slip_uniformity": float(np.max(np.abs(deviation - offset))),
        "delta_phi": peak,
        "diffusivity": diffusivity,
        "source_scheme": source_scheme,
        "converged": steps_run < max_steps,
        "steps_run": steps_run,
    }


def run(case, **params):
    """Dispatch on the case's `experiment` key; return measurements + meta."""
    experiment = params.pop("experiment", case.get("experiment", "dispersion"))
    numerics = dict(case.get("numerics", {}))
    numerics.update(params)

    if experiment == "dispersion":
        lattice = numerics.get("lattice", "D1Q3")
        tau = float(numerics["tau"])
        u = float(numerics.get("u", 0.0))
        order = int(numerics.get("equilibrium_order", 1))
        omega = numerics.get("omega")
        n_cells = int(numerics.get("n_cells", 1024))
        steps = int(numerics.get("steps", 1000))
        sigma0 = float(numerics.get("sigma0", 30.0))
        out = _run_dispersion(lattice, tau, u, order, n_cells, steps, sigma0, omega)
        meta = {
            "solver": "lbm",
            "experiment": experiment,
            "lattice": lattice,
            "tau": tau,
            "u": u,
            "mach": u / np.sqrt(out["cs2"]),
            "equilibrium_order": order,
            "omega": omega,
            "n_cells": n_cells,
            "steps": steps,
            "sigma0": sigma0,
        }
    elif experiment == "dirichlet_slip":
        tau = float(numerics["tau"])
        n_cells = int(numerics.get("n_cells", 32))
        out = _run_dirichlet_slip(
            tau,
            n_cells,
            float(numerics.get("source", 1e-4)),
            float(numerics.get("phi_wall", 1.0)),
            float(numerics.get("steady_tol", 1e-13)),
            int(numerics.get("max_steps", 400000)),
            source_scheme=numerics.get("source_scheme", "corrected"),
        )
        meta = {
            "solver": "lbm",
            "experiment": experiment,
            "lattice": "D1Q3",
            "tau": tau,
            "magic_lambda": lb.magic_lambda(tau),
            "zero_slip_tau": lb.zero_slip_tau(),
            "n_cells": n_cells,
            "source_scheme": numerics.get("source_scheme", "corrected"),
        }
    else:
        raise ValueError(f"unknown experiment {experiment!r}")

    out["meta"] = meta
    return out
