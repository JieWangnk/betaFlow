"""Overdamped Langevin particle runner — the FIRST non-OpenFOAM runner.

Architecture note. This module exists as much to test the harness as to do
physics: betaflow's design constraint has always been "nothing above
runners/ may know OpenFOAM exists", and until now that was asserted rather
than tested, because every runner WAS OpenFOAM. Adding this one exercises the
claim.

PHYSICS. Free Brownian motion, no flow, no walls. The overdamped (inertialess)
limit drops the particle acceleration term, which is justified when the
Stokes number is small: for a 50 nm particle in plasma St ~ 1e-9 (reported in
meta as `stokes_number`), so the velocity relaxation time zeta/m is many
orders below any flow timescale.

The overdamped Langevin equation with no drift is dx = sqrt(2 D) dW, and the
Euler-Maruyama increment is

    dx_i = sqrt(2 D dt) * N(0,1)     per Cartesian component

with D = k_B T / (6 pi mu a) from betaflow.analytic.brownian, so the noise
amplitude and the Stokes drag share one friction coefficient by construction
(fluctuation-dissipation; see that module for the bug this prevents).

EXACTNESS. For free diffusion this scheme is EXACT, not first-order: the true
increment over dt is Gaussian with variance 2 D dt, which is precisely what is
sampled, and sums of independent Gaussians are Gaussian. So the MSD is 6 D t
in expectation for ANY dt. There is therefore no timestep refinement study
here — the convergence axis is PARTICLE COUNT, and the error is statistical
(1/sqrt(N)), not a discretisation error.
"""

import numpy as np

from betaflow.analytic import brownian


def run(case, n_particles=10000, n_steps=100, total_time=None, seed=0, dt=None):
    """Simulate free Brownian motion and return the standard result dict.

    Parameters
    ----------
    case : dict
        Parsed YAML case definition (particle radius, temperature, viscosity).
    n_particles : int
        Ensemble size. This is the convergence axis: the statistical error of
        the MSD slope falls as 1/sqrt(n_particles).
    n_steps, total_time, dt : int, float, float
        Give any two; dt defaults to total_time/n_steps.
    seed : int
        RNG seed, so the regression suite is reproducible.
    """
    phys = case["physical"]
    a = float(phys["particle_radius"])
    temperature = float(phys["temperature"])
    mu = float(phys["dynamic_viscosity"])
    rho_p = float(phys.get("particle_density", 1050.0))

    diffusivity = brownian.stokes_einstein(temperature, mu, a)
    total_time = float(total_time if total_time is not None else phys["total_time"])
    if dt is None:
        dt = total_time / n_steps
    else:
        n_steps = int(round(total_time / dt))

    rng = np.random.default_rng(seed)
    step_sigma = np.sqrt(2.0 * diffusivity * dt)

    x = np.zeros((int(n_particles), 3))
    t = np.zeros(n_steps + 1)
    msd = np.zeros(n_steps + 1)
    msd_components = np.zeros((n_steps + 1, 3))
    for k in range(1, n_steps + 1):
        x += step_sigma * rng.standard_normal(x.shape)
        t[k] = k * dt
        msd_components[k] = np.mean(x**2, axis=0)
        msd[k] = msd_components[k].sum()

    return {
        "t": t,
        "msd": msd,
        "msd_components": msd_components,
        "D_expected": diffusivity,
        "positions": x,
        "meta": {
            "solver": "langevin",
            "scheme": "Euler-Maruyama, overdamped (exact for free diffusion)",
            "n_particles": int(n_particles),
            "n_steps": int(n_steps),
            "dt": float(dt),
            "total_time": float(total_time),
            "seed": int(seed),
            "diffusivity": diffusivity,
            "friction_coefficient": brownian.friction_coefficient(mu, a),
            # Justifies dropping inertia; u_ref/l_ref are nominal blood-flow
            # scales recorded only for this estimate.
            "stokes_number": brownian.stokes_number(rho_p, a, mu, 0.3, 0.01),
            "rms_displacement": float(np.sqrt(msd[-1])),
        },
    }
