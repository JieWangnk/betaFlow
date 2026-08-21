"""OpenLB adapter: the mc_channel case as a D3Q7 Eulerian scalar.

The ONLY betaflow module that knows OpenLB exists. It builds and runs the
C++ app in openlb_cases/mcChannel3d (adapted from OpenLB 1.9's
advectionDiffusion3d example), parses its CSV, and returns the same
`receivers` shape as the Langevin and OpenFOAM particle CIR modes, so the
metrics and tests above this layer do not change.

PARAMETER SELECTION IS BY PREDICTION, NOT TRIAL. With diffusive scaling the
lattice velocity obeys

    u_lat = (tau - 1/2) c_s^2 Pe_cell,      Pe_cell = u_max dx / D

so at Pe = 200 any tau comfortably above 1/2 is UNSTABLE at affordable
resolutions (measured: u_lat = 1.67 at tau = 0.6, res 6 — the field
diverged). Stability pins tau against 1/2 — the corner where the naive
D = c_s^2 tau law is wrong by ~40x and only the exact eigenvalue law
D = c_s^2 (tau - 1/2) survives, which is precisely the exam betaflow's
lattice_boltzmann reference was built to referee. The runner therefore
takes a target u_lat (default 0.04) and computes tau from it, and records
the predicted first-order-equilibrium depletion at that point:
D_eff = (c_s^2 - u_lat(r)^2)(tau - 1/2), radially varying, relative size
u_lat^2/c_s^2 (0.64% at the default — predicted BEFORE the run).

MEASURED INSTRUMENTATION FINDINGS the record carries (openlb_cases/
mcChannel3d/mcChannel3d.cpp has the mechanism comments):
  - momenta::setDensity does not reach BounceBack wall cells; their default
    unit density bled a 2.5x mass growth until iniEquilibrium was used.
  - No density functor on bounce-back cells is mass accounting: stock
    BounceBack reads a FIXED 1 (momenta::FixedDensity), and
    BounceBackBulkDensity reads the Revert collision's period-2 cycle.
    The CIR therefore uses bulk-only sums for numerator and denominator,
    and the mass parked in wall cells mid-flight appears as a slow decline
    of the bulk total — the mass gate bounds it and names the mechanism.
"""

import subprocess
from pathlib import Path

import numpy as np

CS2_D3Q7 = 0.25  # source-confirmed: descriptor cs2<3,7> = {1,4}

_REPO = Path(__file__).resolve().parents[2]
_APP_DIR = _REPO / "openlb_cases" / "mcChannel3d"


def scope_parameters(case, resolution, u_lat_target):
    """(tau, dt, predictions) from the stability law, before any run."""
    from betaflow.analytic import lattice_boltzmann as lb

    phys = case["physical"]
    a = float(phys["vessel_radius"])
    u_max = 2.0 * float(phys["mean_velocity"])
    D = float(phys["diffusivity"])
    dx = a / resolution
    peclet_cell = u_max * dx / D
    tau = 0.5 + u_lat_target / (CS2_D3Q7 * peclet_cell)
    dt = u_lat_target * dx / u_max
    return tau, dt, {
        "peclet_cell": peclet_cell,
        "u_lat_centreline": u_lat_target,
        "tau": tau,
        "naive_over_exact_D_factor": tau / (tau - 0.5),
        # First-order ADE equilibrium (source-confirmed in OpenLB):
        # D_eff = (c_s^2 - u^2)(tau - 1/2), radially varying under
        # Poiseuille. Quoted relative to the requested D at the centreline.
        "depletion_relative_centreline": u_lat_target**2 / CS2_D3Q7,
        "d_lattice_exact": lb.diffusivity(tau, "D3Q7"),
    }


def _build(app="mcChannel3d"):
    app_dir = _REPO / "openlb_cases" / app
    if not (app_dir / app).is_file():
        proc = subprocess.run(["make"], cwd=app_dir, capture_output=True,
                              text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenLB app build failed:\n{proc.stdout[-2000:]}"
                f"\n{proc.stderr[-2000:]}")
    return app_dir / app


def run(case, **params):
    """Dispatch on the case shape, as every other runner does.

    A `receiver` block selects the CIR (scalar) mode; a pipe geometry
    selects the momentum mode — the SAME pipe_poiseuille_steady.yaml that
    examines OpenFOAM examines OpenLB's fluid solver here.
    """
    if "receiver" in case:
        return _run_cir(case, **params)
    if case.get("geometry", {}).get("type") == "pipe":
        return _run_pipe_momentum(case, **params)
    raise ValueError("openlb runner serves the CIR case (receiver block) "
                     "and the pipe momentum case (geometry.type == pipe)")


# Momentum lattice: D3Q19, c_s^2 = 1/3.
CS2_D3Q19 = 1.0 / 3.0


def _run_pipe_momentum(case, resolution=41, tau=0.53, wall="bb",
                       max_phys_t=120.0, workdir=None):
    """Steady forced Poiseuille through openlb_cases/pipeFlow3d.

    Fixed tau across resolutions ON PURPOSE: bounce-back's wall position is
    tau-dependent, so a tau that varies with N would conflate the two
    scalings; at fixed tau the lattice velocity halves per refinement and
    stays stable (u_lat = (tau - 1/2) c_s^2 u_max dx / nu). The stability
    guard raises rather than silently re-scoping.

    Returns the standard fluid-runner shape {y, u, u_ref, meta}, so the
    case's own L2 metric and Re definition-agreement check apply unchanged.
    """
    a = float(case["geometry"]["radius"])
    re_bulk = float(case["nondim"]["Re"])
    # The app fixes u_max = 1, nu from the bulk-Re definition; assert the
    # translation here so a drift in either place fails loudly.
    u_max = 1.0
    u_mean = u_max / 2.0                      # pipe factor of two
    nu = u_mean * (2.0 * a) / re_bulk
    re_cell = u_max * (2.0 * a / resolution) / nu
    u_lat = (tau - 0.5) * CS2_D3Q19 * re_cell
    if u_lat > 0.15:
        raise ValueError(
            f"u_lat = {u_lat:.3f} at N={resolution}, tau={tau}: unstable "
            f"territory; lower tau or raise resolution")

    binary = _build("pipeFlow3d")
    outdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    outdir = outdir / f"pipe_openlb_N{resolution}_tau{tau:g}_{wall}"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [str(binary), "--resolution", str(resolution), "--tau", repr(tau),
         "--wall", wall, "--maxt", repr(float(max_phys_t)),
         "--outdir", str(outdir) + "/"],
        cwd=binary.parent, capture_output=True, text=True)
    if proc.returncode != 0 or "betaflow-done" not in proc.stdout:
        raise RuntimeError(
            f"OpenLB pipe run failed (rc {proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    prov = {}
    for line in proc.stdout.splitlines():
        if "betaflow-provenance" in line:
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    try:
                        prov[k] = float(v)
                    except ValueError:
                        prov[k] = v
    omega = float(prov.get("omega", 0.0))
    tau_realised = 1.0 / omega if omega else float("nan")
    if abs(tau_realised - tau) > 1e-6:
        raise RuntimeError(
            f"OpenLB converter realised tau {tau_realised!r} against "
            f"requested {tau!r}; investigate before trusting the run")

    data = np.loadtxt(outdir / "profile.csv", delimiter=",", skiprows=1)
    y_over_a, u = data[:, 0], data[:, 1]

    return {
        "y": y_over_a * a,
        "u": u,
        # No "tau_w" key on purpose: this mode measures the PROFILE only.
        # A wall-shear number would be the analytic value dressed up as a
        # measurement — staircase walls have no honest local traction here.
        "u_ref": u_max,   # the analytic peak for the applied force at r = a
        "meta": {
            "solver": "openlb",
            "app": "openlb_cases/pipeFlow3d/pipeFlow3d.cpp",
            "mode": f"steady forced Poiseuille, D3Q19 ForcedBGK, wall={wall}",
            "u_mean": u_mean,
            "nu": nu,
            "re_bulk_case_convention": u_mean * 2.0 * a / nu,
            "re_charU_openlb_convention": u_max * 2.0 * a / nu,
            "tau": tau,
            "tau_realised": tau_realised,
            "wall": wall,
            "resolution_per_diameter": int(resolution),
            "dx": 2.0 * a / resolution,
            "u_lat_char": u_lat,
            "max_phys_t": float(max_phys_t),
            "provenance_from_app": prov,
        },
    }


def _run_cir(case, resolution=12, u_lat_target=0.04, time_horizon_over_t2=6.5,
             outputs=400, workdir=None):
    """Build, execute, parse; the case supplies physics, this layer numerics."""
    from betaflow.analytic import channel_impulse as ci

    binary = _build()
    tau, dt, predictions = scope_parameters(case, resolution, u_lat_target)

    outdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    outdir = outdir / f"mc_channel_openlb_res{resolution}_u{u_lat_target:g}"
    outdir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [str(binary),
         "--resolution", str(resolution),
         "--tau", repr(tau),
         "--horizon", repr(float(time_horizon_over_t2)),
         "--outputs", str(outputs),
         "--outdir", str(outdir) + "/"],
        cwd=_APP_DIR, capture_output=True, text=True)
    if proc.returncode != 0 or "betaflow-done" not in proc.stdout:
        raise RuntimeError(
            f"OpenLB run failed (rc {proc.returncode}):\n"
            f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")

    prov = {}
    for line in proc.stdout.splitlines():
        if "betaflow-provenance" in line:
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    prov[k] = float(v) if v.replace(".", "").replace(
                        "-", "").replace("e", "").isdigit() else v
    # The app derives omega through OpenLB's own unit converter; it must
    # recover the requested tau or the converter disagrees with the law
    # this runner used — that would be a finding, so it is checked.
    omega = float(prov.get("omega", 0.0))
    tau_realised = 1.0 / omega if omega else float("nan")
    if abs(tau_realised - tau) > 1e-6:
        raise RuntimeError(
            f"OpenLB unit converter realised tau {tau_realised!r} against "
            f"requested {tau!r} — scaling laws disagree; investigate before "
            f"trusting any number from this run")

    data = np.loadtxt(outdir / "cir.csv", delimiter=",", skiprows=1)
    t = data[:, 0]
    mass = data[:, 4]

    phys = case["physical"]
    u_mean = float(phys["mean_velocity"])
    c_x = float(case["receiver"]["axial_length"])
    distances = [float(d) for d in case["receiver"]["distances"]]
    slug_w = float(prov.get("slugW", 0.0))

    receivers = []
    for k, d in enumerate(distances):
        # Reference averaged over the finite slug width (exact quadrature
        # of the closed form): the app releases a slug, not a delta.
        ss = np.linspace(-slug_w / 2.0, slug_w / 2.0, 21)
        ref = np.mean([ci.cir(t, u_mean, d + s, c_x) for s in ss], axis=0)
        receivers.append({
            "dbar": d,
            "t": t,
            "cir_measured": data[:, 1 + k],
            "cir_reference": ref,
            "t1": ci.onset_time(u_mean, d, c_x),
            "t2": ci.peak_time(u_mean, d, c_x),
            "peak_value": ci.peak_value(d, c_x),
            "flow_dominated_ratio": ci.flow_dominated(
                u_mean * float(phys["vessel_radius"])
                / float(phys["diffusivity"]),
                d, float(phys["vessel_radius"])),
        })

    return {
        "receivers": receivers,
        "mass_over_initial": mass / mass[0],
        "meta": {
            "solver": "openlb",
            "app": str(_APP_DIR / "mcChannel3d.cpp"),
            "openlb_version": "1.9.0",
            "descriptor": "D3Q7<VELOCITY>, AdvectionDiffusionBGKdynamics "
                          "(first-order equilibrium, source-confirmed)",
            "mode": "cir: Eulerian slug, prescribed Poiseuille, "
                    "bounce-back walls, bulk-only accounting",
            "resolution_cells_per_radius": int(resolution),
            "predictions_before_run": predictions,
            "provenance_from_app": prov,
            "tau_realised": tau_realised,
            "dt": dt,
            "outputs": int(outputs),
            "time_horizon_over_t2": float(time_horizon_over_t2),
        },
    }
