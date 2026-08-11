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


def _build():
    if not (_APP_DIR / "mcChannel3d").is_file():
        proc = subprocess.run(["make"], cwd=_APP_DIR, capture_output=True,
                              text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenLB app build failed:\n{proc.stdout[-2000:]}"
                f"\n{proc.stderr[-2000:]}")
    return _APP_DIR / "mcChannel3d"


def run(case, resolution=12, u_lat_target=0.04, time_horizon_over_t2=6.5,
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
