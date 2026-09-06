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


def _build(app="mcChannel3d", binary=None):
    """Build (if absent) and return the app binary. `binary` covers the
    directories whose binary name differs from the directory name."""
    app_dir = _REPO / "openlb_cases" / app
    binary = binary or app
    if not (app_dir / binary).is_file():
        proc = subprocess.run(["make"], cwd=app_dir, capture_output=True,
                              text=True)
        if proc.returncode != 0:
            raise RuntimeError(
                f"OpenLB app build failed:\n{proc.stdout[-2000:]}"
                f"\n{proc.stderr[-2000:]}")
    return app_dir / binary


def run(case, **params):
    """Dispatch on the case shape, as every other runner does.

    A `receiver` block selects the CIR (scalar) mode; a pipe geometry
    selects the momentum mode — the SAME pipe_poiseuille_steady.yaml that
    examines OpenFOAM examines OpenLB's fluid solver here.
    """
    if params.pop("junction", False):
        return _run_junction(case, **params)
    if "bend_angle_deg" in params:
        return _run_bent(case, **params)
    if "receiver" in case:
        return _run_cir(case, **params)
    if case.get("geometry", {}).get("type") == "pipe":
        return _run_pipe_momentum(case, **params)
    raise ValueError("openlb runner serves the CIR case (receiver block), "
                     "the pipe momentum case (geometry.type == pipe), and "
                     "the bent-pipe G4 control (bend_angle_deg param)")


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


def _run_junction(case, resolution=12, u_lat_target=0.04,
                  time_horizon_over_t2=6.5, outputs=400, workdir=None,
                  magic_lambda=0.25, fluid_maxt=3.0):
    """The pre-registered Y-junction: solved flow (junctionFlow3d), then
    the NT+TRT scalar (junctionPipe3d) on the frozen field. Five windows:
    the mother in-run control and a mirror pair per daughter — gate G3
    compares the pairs on the same deterministic solve. There is no
    prescribed-velocity mode: no analytic junction field exists.
    """
    from betaflow.analytic import channel_impulse as ci

    fluid_bin = _build("junction3d", "junctionFlow3d")
    scalar_bin = _build("junctionScalar3d", "junctionPipe3d")
    tau, dt, predictions = scope_parameters(case, resolution, u_lat_target)
    tau_even = magic_lambda / (tau - 0.5) + 0.5

    base = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    fdir = base / f"junction_flow_res{resolution}_t{fluid_maxt:g}"
    fdir.mkdir(parents=True, exist_ok=True)
    fprov = fdir / "provenance.txt"
    if not (fprov.is_file() and (fdir / "ufield.csv").is_file()):
        proc = subprocess.run(
            [str(fluid_bin),
             "--resolution", str(resolution),
             "--horizon", repr(float(time_horizon_over_t2)),
             "--maxt", repr(float(fluid_maxt)),
             "--outdir", str(fdir) + "/"],
            cwd=fluid_bin.parent, capture_output=True, text=True)
        if proc.returncode != 0 or "betaflow-done" not in proc.stdout:
            raise RuntimeError(
                f"junctionFlow3d failed:\n{proc.stdout[-2000:]}"
                f"\n{proc.stderr[-2000:]}")
        fprov.write_text("\n".join(
            ln for ln in proc.stdout.splitlines()
            if "betaflow-" in ln) + "\n")
    gates = {}
    for line in fprov.read_text().splitlines():
        for tok in line.split():
            if "=" in tok:
                k, v = tok.split("=", 1)
                try:
                    gates[k] = float(v)
                except ValueError:
                    gates[k] = v
    prof = np.genfromtxt(fdir / "profiles.csv", delimiter=",",
                         skip_header=1, dtype=None, encoding="utf8")
    u_mean = float(case["physical"]["mean_velocity"])
    murray = 2.0 ** (-1.0 / 3.0)
    fluid_meta = {"gates": gates}
    for st, umax in (("mother", 2.0 * u_mean),
                     ("daughter_plus", 2.0 * u_mean * murray),
                     ("daughter_minus", 2.0 * u_mean * murray)):
        y = np.array([r[1] for r in prof if r[0].strip() == st])
        u = np.array([r[2] for r in prof if r[0].strip() == st])
        fluid_meta[f"{st}_L2_vs_parabola"] = float(np.sqrt(np.mean(
            (u / umax - (1.0 - y**2))**2)))
    qm = gates.get("flux_mother")
    qp = gates.get("flux_daughter_plus")
    qn = gates.get("flux_daughter_minus")
    if qm:
        fluid_meta["flux_balance"] = (qp + qn) / qm - 1.0
        fluid_meta["flux_split_asymmetry"] = qp / qn - 1.0

    sdir = base / f"junction_scalar_res{resolution}_u{u_lat_target:g}_p3"
    sdir.mkdir(parents=True, exist_ok=True)
    sprov = sdir / "provenance.txt"
    reused = (sprov.is_file() and (sdir / "cir.csv").is_file()
              and sum(1 for _ in open(sdir / "cir.csv")) >= outputs)
    if not reused:
        proc = subprocess.run(
            [str(scalar_bin),
             "--resolution", str(resolution),
             "--tau", repr(tau),
             "--taueven", repr(tau_even),
             "--horizon", repr(float(time_horizon_over_t2)),
             "--outputs", str(outputs),
             "--ufield", str(fdir / "ufield.csv"),
             "--outdir", str(sdir) + "/"],
            cwd=scalar_bin.parent, capture_output=True, text=True)
        if proc.returncode != 0 or "betaflow-done" not in proc.stdout:
            raise RuntimeError(
                f"junctionPipe3d failed:\n{proc.stdout[-2000:]}"
                f"\n{proc.stderr[-2000:]}")
        sprov.write_text("\n".join(
            ln for ln in proc.stdout.splitlines()
            if "betaflow-" in ln) + "\n")

    data = np.loadtxt(sdir / "cir.csv", delimiter=",", skiprows=1)
    t = data[:, 0]
    mass = data[:, 12]
    c_x = float(case["receiver"]["axial_length"])
    dists = [float(d) for d in case["receiver"]["distances"]]
    sweep_um = (600, 750, 1000, 1250, 1550)

    def straight_ref(d):
        return ci.cir(t, u_mean, d, c_x)

    windows = {
        "mother_150": {"cir": data[:, 1], "dbar": dists[0],
                       "t2": ci.peak_time(u_mean, dists[0], c_x),
                       "ref_straight": straight_ref(dists[0])},
    }
    for k, um in enumerate(sweep_um):
        d = um * 1e-6
        for tag, col in (("plus", 2 + k), ("minus", 7 + k)):
            windows[f"daughter_{tag}_{um}"] = {
                "cir": data[:, col], "dbar": d,
                "t2": ci.peak_time(u_mean, d, c_x),
                "ref_straight": straight_ref(d),
            }

    return {
        "windows": windows,
        "t": t,
        "mass_over_initial": mass / mass[0],
        "meta": {
            "solver": "openlb",
            "apps": [str(fluid_bin) + ".cpp", str(scalar_bin) + ".cpp"],
            "mode": "Y-junction: solved D3Q19 flow (Bouzidi) -> frozen "
                    "field -> D3Q7 NT+TRT scalar; five bulk-material "
                    "windows",
            "resolution_cells_per_radius": int(resolution),
            "predictions_before_run": predictions,
            "fluid_stage": fluid_meta,
            "magic_lambda": float(magic_lambda),
            "outputs": int(outputs),
            "time_horizon_over_t2": float(time_horizon_over_t2),
        },
    }


def _run_bent(case, bend_angle_deg=0.0, resolution=12, u_lat_target=0.04,
              time_horizon_over_t2=6.5, outputs=400, workdir=None,
              dynamics="trt", magic_lambda=0.25, wall="bb",
              velocity="analytic", fluid_maxt=3.0):
    """Gate G4 of the bifurcation pre-registration: the bent-pipe control.

    openlb_cases/bentPipe3d rebuilds the mc_channel scalar transport
    through the junction machinery (composite arc geometry, region-wise
    analytic velocity, path-based windows, capped ends). At angle 0 the
    straight record is the known answer, so deviations are machinery; at
    the pre-registered 30 degrees the deviation FROM THE CONTROL is the
    bend. Prescribed-field leg only — the solved bent flow is a later
    rung, the same staging order the straight case used.

    dynamics defaults to TRT at magic Lambda = 1/4 because stock BGK is
    UNSTABLE for oblique advection past the bounce-back staircase at the
    stability-pinned tau (measured 2026-08-26; the full discriminating
    ladder is in the G4 record and the app header). The TRT's odd rate
    carries the diffusivity, so Lambda touches stability only, never D.
    """
    from betaflow.analytic import channel_impulse as ci

    binary = _build("bentPipe3d")
    tau, dt, predictions = scope_parameters(case, resolution, u_lat_target)

    # velocity="solved": run the D3Q19 fluid stage (bentFlow3d) through the
    # SAME layout first — Bouzidi walls, inlet Poiseuille, outlet pressure,
    # warm-started from the analytic field — then advect the scalar on the
    # frozen solved field. Same staging as the straight coupled model.
    fluid_meta = None
    ufield_arg = []
    if velocity == "solved":
        fluid_bin = _build("bentFlow3d")
        fdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
        fdir = fdir / (f"bent_flow_res{resolution}_a{bend_angle_deg:g}"
                       f"_t{fluid_maxt:g}")
        fdir.mkdir(parents=True, exist_ok=True)
        fprov = fdir / "provenance.txt"
        if not (fprov.is_file() and (fdir / "ufield.csv").is_file()):
            proc = subprocess.run(
                [str(fluid_bin),
                 "--resolution", str(resolution),
                 "--angle", repr(float(bend_angle_deg)),
                 "--horizon", repr(float(time_horizon_over_t2)),
                 "--maxt", repr(float(fluid_maxt)),
                 "--outdir", str(fdir) + "/"],
                cwd=fluid_bin.parent, capture_output=True, text=True)
            if proc.returncode != 0 or "betaflow-done" not in proc.stdout:
                raise RuntimeError(
                    f"bentFlow3d failed:\n{proc.stdout[-2000:]}"
                    f"\n{proc.stderr[-2000:]}")
            fprov.write_text("\n".join(
                ln for ln in proc.stdout.splitlines()
                if "betaflow-" in ln) + "\n")
        gates = {}
        for line in fprov.read_text().splitlines():
            if "betaflow-gates" in line or "betaflow-provenance" in line:
                for tok in line.split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        try:
                            gates[k] = float(v)
                        except ValueError:
                            gates[k] = v
        prof = np.genfromtxt(fdir / "profiles.csv", delimiter=",",
                             skip_header=1, dtype=None, encoding="utf8")
        u_max = 2.0 * float(case["physical"]["mean_velocity"])
        fluid_meta = {"gates": gates}
        for st in ("mother", "daughter"):
            y = np.array([r[1] for r in prof if r[0].strip() == st])
            u = np.array([r[2] for r in prof if r[0].strip() == st])
            fluid_meta[f"{st}_L2_vs_parabola"] = float(np.sqrt(np.mean(
                (u / u_max - (1.0 - y**2))**2)))
        q = [gates.get("flux_mother"), gates.get("flux_daughter_near"),
             gates.get("flux_daughter_far")]
        if all(v is not None for v in q) and q[0]:
            fluid_meta["flux_imbalance_near"] = q[1]/q[0] - 1.0
            fluid_meta["flux_imbalance_far"] = q[2]/q[0] - 1.0
        ufield_arg = ["--ufield", str(fdir / "ufield.csv")]

    tau_even = magic_lambda / (tau - 0.5) + 0.5

    outdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    name = (f"bent_pipe_res{resolution}"
            f"_a{bend_angle_deg:g}_u{u_lat_target:g}_{dynamics}")
    if wall != "bb":
        name += f"_{wall}"
    if velocity == "solved":
        name += "_solved"
    outdir = outdir / name
    outdir.mkdir(parents=True, exist_ok=True)

    # Deterministic reuse (the wall-sweep tool's pattern): the app is a pure
    # function of its CLI, so a complete cir.csv plus its provenance line is
    # the run. Lets the G4 test's three legs resume across invocations.
    prov_file = outdir / "provenance.txt"
    reused = (prov_file.is_file() and (outdir / "cir.csv").is_file()
              and sum(1 for _ in open(outdir / "cir.csv")) >= outputs)
    if reused:
        stdout = prov_file.read_text()
    else:
        proc = subprocess.run(
            [str(binary),
             "--resolution", str(resolution),
             "--tau", repr(tau),
             "--horizon", repr(float(time_horizon_over_t2)),
             "--outputs", str(outputs),
             "--angle", repr(float(bend_angle_deg)),
             "--dynamics", dynamics,
             "--taueven", repr(tau_even),
             "--wall", wall,
             "--outdir", str(outdir) + "/"] + ufield_arg,
            cwd=binary.parent, capture_output=True, text=True)
        if proc.returncode != 0 or "betaflow-done" not in proc.stdout:
            raise RuntimeError(
                f"bentPipe3d run failed (rc {proc.returncode}):\n"
                f"{proc.stdout[-2000:]}\n{proc.stderr[-2000:]}")
        stdout = proc.stdout
        prov_file.write_text("\n".join(
            ln for ln in stdout.splitlines() if "betaflow-" in ln) + "\n")

    prov = {}
    for line in stdout.splitlines():
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
            f"unit converter realised tau {tau_realised!r} against "
            f"requested {tau!r}")

    data = np.loadtxt(outdir / "cir.csv", delimiter=",", skiprows=1)
    t = data[:, 0]
    mass = data[:, 4]

    phys = case["physical"]
    u_mean = float(phys["mean_velocity"])
    c_x = float(case["receiver"]["axial_length"])
    slug_w = float(prov.get("slugW", 0.0))

    receivers = []
    for k, d in enumerate([float(x) for x in case["receiver"]["distances"]]):
        ss = np.linspace(-slug_w / 2.0, slug_w / 2.0, 21)
        ref = np.mean([ci.cir(t, u_mean, d + s, c_x) for s in ss], axis=0)
        receivers.append({
            "dbar": d,
            "t": t,
            "cir_measured": data[:, 1 + k],
            "cir_reference_straight": ref,
            "t2": ci.peak_time(u_mean, d, c_x),
        })

    return {
        "receivers": receivers,
        "mass_over_initial": mass / mass[0],
        "meta": {
            "solver": "openlb",
            "app": str(_REPO / "openlb_cases" / "bentPipe3d"
                       / "bentPipe3d.cpp"),
            "openlb_version": "1.9.0",
            "mode": "bent-pipe G4 control: Eulerian slug, piecewise "
                    "prescribed Poiseuille (mitred), bounce-back walls, "
                    "capped ends, path-based windows, bulk-only accounting",
            "bend_angle_deg": float(bend_angle_deg),
            "wall": wall,
            "velocity_source": velocity,
            "fluid_stage": fluid_meta,
            "dynamics": dynamics,
            "magic_lambda": float(magic_lambda) if dynamics == "trt" else None,
            "tau_even": float(tau_even) if dynamics == "trt" else None,
            "u_lat_target": float(u_lat_target),
            "reused_existing_output": bool(reused),
            "resolution_cells_per_radius": int(resolution),
            "predictions_before_run": predictions,
            "provenance_from_app": prov,
            "tau_realised": tau_realised,
            "dt": dt,
            "outputs": int(outputs),
            "time_horizon_over_t2": float(time_horizon_over_t2),
        },
    }


def _run_cir(case, resolution=12, u_lat_target=0.04, time_horizon_over_t2=6.5,
             outputs=400, workdir=None, coupled=False):
    """Build, execute, parse; the case supplies physics, this layer numerics.

    coupled=True is the COUPLED channel scenario: OpenLB's own fluid
    solver (openlb_cases/pipeFlow3d, Bouzidi walls per the wall-position
    measurement) produces the flow at the case's microfluidic physics
    (Re = u_mean*2a/nu = 0.6, water), and the scalar rides on the SOLVED
    profile instead of the analytic one. The flow is steady, so staging
    (converge fluid, freeze profile, run scalar) is physically identical
    to per-step coupling and decouples the two lattices' time steps —
    which matters because Sc = nu/D = 667 would otherwise force
    tau_fluid ~ 2.9 on a shared step. The fluid stage's own exam numbers
    (profile L2 vs the parabola, effective-radius shift) are returned in
    meta["fluid_stage"].
    """
    from betaflow.analytic import channel_impulse as ci

    binary = _build()

    fluid_meta = None
    profile_arg = []
    if coupled:
        phys = case["physical"]
        a = float(phys["vessel_radius"])
        u_max = 2.0 * float(phys["mean_velocity"])
        nu = float(phys["kinematic_viscosity"])
        fluid_bin = _build("pipeFlow3d")
        fdir = (Path(workdir) if workdir is not None else Path.cwd() / "_runs")
        fdir = fdir / f"mc_channel_fluid_res{resolution}"
        fdir.mkdir(parents=True, exist_ok=True)
        proc = subprocess.run(
            [str(fluid_bin), "--radius", repr(a), "--umax", repr(u_max),
             "--nu", repr(nu), "--resolution", str(2 * resolution),
             "--tau", "0.8", "--wall", "bouzidi", "--maxt", "0.5",
             "--outdir", str(fdir) + "/"],
            cwd=fluid_bin.parent, capture_output=True, text=True)
        if proc.returncode != 0 or "betaflow-done" not in proc.stdout:
            raise RuntimeError(
                f"coupled fluid stage failed:\n{proc.stdout[-1500:]}")
        fp = np.loadtxt(fdir / "profile.csv", delimiter=",", skiprows=1)
        y, u = fp[:, 0], fp[:, 1]
        sel = np.abs(y) <= 0.85
        A = np.column_stack([np.ones(int(sel.sum())), y[sel] ** 2])
        cf, *_ = np.linalg.lstsq(A, u[sel], rcond=None)
        a_eff = float(np.sqrt(-cf[0] / cf[1]))
        fluid_meta = {
            "wall": "bouzidi", "tau": 0.8,
            "resolution_per_diameter": 2 * resolution,
            "reynolds_bulk": float(phys["mean_velocity"]) * 2.0 * a / nu,
            "schmidt": nu / float(phys["diffusivity"]),
            "L2_vs_parabola": float(np.sqrt(np.mean(
                (u / u_max - (1.0 - y**2)) ** 2))),
            "u_max_fit_over_target": float(cf[0] / u_max),
            "a_eff_shift_dx": (a_eff - 1.0) / (2.0 / (2 * resolution)),
            "profile_csv": str(fdir / "profile.csv"),
            "staging_note": "steady flow: fluid converged then frozen; "
                            "per-step coupling adds nothing at steady "
                            "state and a shared dt at Sc = 667 would "
                            "force tau_fluid ~ 2.9",
        }
        profile_arg = ["--profile", str(fdir / "profile.csv")]
    tau, dt, predictions = scope_parameters(case, resolution, u_lat_target)

    outdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    outdir = outdir / (f"mc_channel_openlb_res{resolution}_u{u_lat_target:g}"
                      + ("_coupled" if coupled else ""))
    outdir.mkdir(parents=True, exist_ok=True)

    proc = subprocess.run(
        [str(binary),
         "--resolution", str(resolution),
         "--tau", repr(tau),
         "--horizon", repr(float(time_horizon_over_t2)),
         "--outputs", str(outputs),
         "--outdir", str(outdir) + "/"] + profile_arg,
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
            "fluid_stage": fluid_meta,
        },
    }
