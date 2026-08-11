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


def _specular_reflect(p0, p1, a, max_bounces=4):
    """Specular reflection of a straight step off the cylinder wall r = a.

    NOT naive rejection and NOT radial folding. Rejection (redraw or clamp)
    depletes or piles up the near-wall population and biases D_eff invisibly
    through the velocity sampling — the most likely silent error in this case.
    Radial folding r -> 2a - r is measure-distorting in 2-D, because the area
    element 2 pi r dr differs between r and 2a - r.

    Here the crossing point is solved exactly for the straight segment,
    |p0 + t d| = a, the wall normal is n = crossing/a, and the REMAINING part
    of the step is mirrored: d' = d - 2(d.n)n. Iterated for the rare step that
    re-exits, which needs a step comparable to the radius and is excluded by
    the timestep constraint anyway.
    """
    p0 = p0.copy()
    p1 = p1.copy()
    for _ in range(max_bounces):
        outside = np.einsum("ij,ij->i", p1, p1) > a * a
        if not outside.any():
            break
        q0, q1 = p0[outside], p1[outside]
        d = q1 - q0
        qa = np.einsum("ij,ij->i", d, d)
        qb = 2.0 * np.einsum("ij,ij->i", q0, d)
        qc = np.einsum("ij,ij->i", q0, q0) - a * a
        disc = np.maximum(qb * qb - 4.0 * qa * qc, 0.0)
        t = np.clip((-qb + np.sqrt(disc)) / np.maximum(2.0 * qa, 1e-300), 0.0, 1.0)
        crossing = q0 + t[:, None] * d
        n = crossing / a
        remaining = (1.0 - t)[:, None] * d
        reflected = remaining - 2.0 * np.einsum("ij,ij->i", remaining, n)[:, None] * n
        p0[outside] = crossing
        p1[outside] = crossing + reflected
    return p1


def _run_pipe(case, n_particles, seed, epsilon, cycles, particle_radius):
    """Overdamped Langevin in a cylinder with axial Poiseuille flow.

    ONE-WAY COUPLING ONLY: the particles are advected by a prescribed
    analytic velocity field and do not perturb it. Justified by the dilute,
    St << 1 single-particle regime these models assume (reported in meta as
    stokes_number; the review assumptions are recorded in the case YAML).

    Timestep and duration follow from two constraints, both reported:
      radial step ratio  sqrt(2 D dt)/a = epsilon
      duration           t_max = cycles * tau_r,  tau_r = a^2/D
    so n_steps = 2*cycles/epsilon^2 — independent of the physical scales,
    which is why the radius sweep costs the same per point.
    """
    from betaflow.analytic import taylor_aris as ta

    phys = case["physical"]
    a = float(phys["vessel_radius"])
    u_mean = float(phys["mean_velocity"])
    temperature = float(phys["temperature"])
    mu = float(phys["dynamic_viscosity"])
    rho_p = float(phys.get("particle_density", 1050.0))
    r_p = float(particle_radius if particle_radius is not None else phys["particle_radius"])

    diffusivity = brownian.stokes_einstein(temperature, mu, r_p)
    tau_r = ta.radial_relaxation_time(a, diffusivity)
    dt = epsilon**2 * tau_r / 2.0
    n_steps = int(round(2.0 * cycles / epsilon**2))

    rng = np.random.default_rng(seed)
    sigma_step = np.sqrt(2.0 * diffusivity * dt)
    n = int(n_particles)

    # Start uniform in the cross-section, which is the equilibrium P(r)=2r/a^2:
    # sqrt of a uniform variate gives exactly that radial law.
    radius0 = a * np.sqrt(rng.random(n))
    theta0 = 2.0 * np.pi * rng.random(n)
    p = np.column_stack((radius0 * np.cos(theta0), radius0 * np.sin(theta0)))
    x = np.zeros(n)

    t = np.zeros(n_steps + 1)
    var_x = np.zeros(n_steps + 1)
    ks_times, ks_stats = [], []
    from scipy.stats import kstest

    report_every = max(1, n_steps // 8)
    for k in range(1, n_steps + 1):
        r_now = np.hypot(p[:, 0], p[:, 1])
        u_axial = ta.velocity_profile(r_now / a, u_mean)
        x += u_axial * dt + sigma_step * rng.standard_normal(n)
        p_new = p + sigma_step * rng.standard_normal((n, 2))
        p = _specular_reflect(p, p_new, a)
        t[k] = k * dt
        var_x[k] = np.var(x)
        if k % report_every == 0:
            r_k = np.hypot(p[:, 0], p[:, 1]) / a
            ks_times.append(t[k] / tau_r)
            ks_stats.append(float(kstest(r_k, ta.radial_cdf).statistic))

    r_final = np.hypot(p[:, 0], p[:, 1]) / a
    return {
        "t": t,
        "var_x": var_x,
        "r_over_a": r_final,
        "ks_statistic": float(kstest(r_final, ta.radial_cdf).statistic),
        "ks_history": {"t_over_tau_r": ks_times, "statistic": ks_stats},
        "D_expected": diffusivity,
        "D_eff_expected": ta.d_eff(diffusivity, a, u_mean),
        "meta": {
            "solver": "langevin",
            "mode": "cylinder + analytic Poiseuille, one-way coupled",
            "n_particles": n,
            "n_steps": n_steps,
            "dt": float(dt),
            "epsilon_radial_step": float(epsilon),
            "cycles_tau_r": float(cycles),
            "t_over_tau_r": float(t[-1] / tau_r),
            "tau_r": float(tau_r),
            "seed": int(seed),
            "particle_radius": r_p,
            "diffusivity": diffusivity,
            "peclet": float(a * u_mean / diffusivity),
            "stokes_number": brownian.stokes_number(rho_p, r_p, mu, u_mean, a),
            "wall_scheme": "specular reflection, exact crossing, iterated",
        },
    }


def _run_cir(case, n_particles, seed, epsilon, diffusivity=None,
             time_horizon_over_t2=None):
    """Channel impulse response: uniform release, transparent receivers.

    The molecular-communications measurement (Hofmann et al. 2024 setup, see
    betaflow/analytic/channel_impulse.py): particles released at t = 0
    uniformly over the cross-section at x = 0, advected by Poiseuille flow,
    counted inside axial windows [dbar - c_x/2, dbar + c_x/2]. ONE release
    serves every receiver, because the receivers are transparent.

    Two regimes, selected by the diffusivity:

    D == 0 — EXACT KINEMATICS. Each particle keeps its release radius, so
    x(t) = u(r0) t in closed form: no time-stepping, no scheme error, and the
    ONLY departure from the analytic reference is the finite-N draw of the release
    positions, whose law is binomial (metrics/mc_error.py). Counting uses a
    sort of the release speeds: x in [lo, hi] iff u0 in [lo/t, hi/t].

    D > 0 — the departure measurement. Euler-Maruyama with axial noise,
    radial noise, and the same specular wall reflection as the dispersion
    case. Measured departures from the flow-dominated analytic reference (recorded in
    results/mc_channel_departure.json): axial diffusion softens the onset
    (arrivals before t1) and depresses the peak, and the tail departs in TWO
    regimes, in the OPPOSITE order to the naive expectation. The prediction
    written here before measuring was "radial diffusion walks slow near-wall
    particles onto fast streamlines, so the tail falls BELOW the analytic reference" —
    wrong at intermediate times, because the same mechanism pumps the
    upstream reservoir of still-slower particles INTO the window, and that
    reservoir outweighs the window population (mass ratio ~ dbar/c_x). So
    the measured tail is first ENHANCED (up to 1.6x at 5 t2), then crosses
    below and TERMINATES: exactly zero by 12 t2 at the middle receiver,
    where the analytic reference's log-divergent tail still predicts 1e-2. The measured
    crossover, 0.065 tau_r at one seed and one parameter point, sits at
    0.95 tau_r/beta_1^2 — consistent with the radial relaxation EIGENTIME
    tau_r/beta_1^2 (beta_1 = 3.8317), recorded as a hypothesis, not a law.
    The radial invariant P(r) = 2r/a^2 stays exact and is this mode's gate.
    """
    from betaflow.analytic import channel_impulse as ci
    from betaflow.analytic import taylor_aris as ta

    phys = case["physical"]
    recv = case["receiver"]
    num = case.get("numerics", {})
    a = float(phys["vessel_radius"])
    u_mean = float(phys["mean_velocity"])
    c_x = float(recv["axial_length"])
    distances = [float(d) for d in recv["distances"]]
    d_diff = float(phys.get("diffusivity", 0.0) if diffusivity is None
                   else diffusivity)
    horizon = float(num.get("time_horizon_over_t2", 10.0)
                    if time_horizon_over_t2 is None else time_horizon_over_t2)
    n_samples = int(num.get("n_samples", 1500))
    resolution_steps = int(num.get("cir_resolution_steps", 50))

    rng = np.random.default_rng(seed)
    n = int(n_particles)
    radius0 = a * np.sqrt(rng.random(n))
    theta0 = 2.0 * np.pi * rng.random(n)
    p = np.column_stack((radius0 * np.cos(theta0), radius0 * np.sin(theta0)))

    t1 = {d: ci.onset_time(u_mean, d, c_x) for d in distances}
    t2 = {d: ci.peak_time(u_mean, d, c_x) for d in distances}
    receivers = []
    meta = {
        "solver": "langevin",
        "mode": "cir: uniform release, analytic Poiseuille, transparent receivers",
        "n_particles": n,
        "seed": int(seed),
        "diffusivity": d_diff,
        "time_horizon_over_t2": horizon,
        "vessel_radius": a,
        "mean_velocity": u_mean,
        "receiver_length": c_x,
    }

    if d_diff == 0.0:
        # Exact kinematics: per-receiver grid in t/t2 units so each rise is
        # equally resolved, counts by binary search on the sorted speeds.
        u_sorted = np.sort(ta.velocity_profile(radius0 / a, u_mean))
        for d in distances:
            lo, hi = d - c_x / 2.0, d + c_x / 2.0
            tt = np.linspace(0.5 * t1[d], horizon * t2[d], n_samples)
            counts = (np.searchsorted(u_sorted, hi / tt, side="right")
                      - np.searchsorted(u_sorted, lo / tt, side="left"))
            receivers.append({
                "dbar": d,
                "t": tt,
                "cir_measured": counts / n,
                "cir_reference": ci.cir(tt, u_mean, d, c_x),
                "t1": t1[d],
                "t2": t2[d],
                "peak_value": ci.peak_value(d, c_x),
            })
        meta["scheme"] = "closed-form x = u(r0) t; only the release draw is sampled"
    else:
        # Time-stepped departure run. dt resolves BOTH the radial walk
        # (epsilon = sqrt(2 D dt)/a, the dispersion case's constraint) and
        # the fastest receiver's rise (t2_min / resolution_steps) — at high
        # Peclet the second one binds.
        tau_r = ta.radial_relaxation_time(a, d_diff)
        dt = min(epsilon**2 * tau_r / 2.0, min(t2.values()) / resolution_steps)
        t_max = horizon * max(t2.values())
        n_steps = int(np.ceil(t_max / dt))
        sigma_step = np.sqrt(2.0 * d_diff * dt)

        x = np.zeros(n)
        tt = np.arange(1, n_steps + 1) * dt
        fractions = {d: np.zeros(n_steps) for d in distances}
        windows = {d: (d - c_x / 2.0, d + c_x / 2.0) for d in distances}
        for k in range(n_steps):
            r_now = np.hypot(p[:, 0], p[:, 1])
            x += (ta.velocity_profile(r_now / a, u_mean) * dt
                  + sigma_step * rng.standard_normal(n))
            p_new = p + sigma_step * rng.standard_normal((n, 2))
            p = _specular_reflect(p, p_new, a)
            for d, (lo, hi) in windows.items():
                fractions[d][k] = np.count_nonzero((x >= lo) & (x <= hi)) / n
        for d in distances:
            receivers.append({
                "dbar": d,
                "t": tt,
                "cir_measured": fractions[d],
                "cir_reference": ci.cir(tt, u_mean, d, c_x),
                "t1": t1[d],
                "t2": t2[d],
                "peak_value": ci.peak_value(d, c_x),
                "flow_dominated_ratio": ci.flow_dominated(
                    u_mean * a / d_diff, d, a),
            })
        meta.update({
            "scheme": "Euler-Maruyama, axial + radial noise, specular wall",
            "dt": float(dt),
            "n_steps": n_steps,
            "epsilon_radial_step": float(np.sqrt(2.0 * d_diff * dt) / a),
            "tau_r": float(tau_r),
            "t_max_over_tau_r": float(t_max / tau_r),
            "peclet": float(u_mean * a / d_diff),
        })

    r_final = np.hypot(p[:, 0], p[:, 1]) / a
    from scipy.stats import kstest

    return {
        "receivers": receivers,
        "r_over_a": r_final,
        "ks_statistic": float(kstest(r_final, ta.radial_cdf).statistic),
        "meta": meta,
    }


def run(case, n_particles=10000, n_steps=100, total_time=None, seed=0, dt=None,
        epsilon=0.05, cycles=10.0, particle_radius=None, diffusivity=None,
        time_horizon_over_t2=None):
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
    if "receiver" in case:
        return _run_cir(case, n_particles, seed, epsilon, diffusivity,
                        time_horizon_over_t2)
    if "flow" in case:
        return _run_pipe(case, n_particles, seed, epsilon, cycles, particle_radius)

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
