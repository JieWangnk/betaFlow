"""OpenFOAM 14 (Foundation) adapter — the only module that knows OpenFOAM exists.

All case types share the same box mesh: one cell thick in z with `empty`
front/back patches, cyclic streamwise patches, and separate bottomWall /
topWall patches. What differs per case type is the DRIVING MECHANISM, split
out in _SETUPS:

  channel — walls at y = ±h, both no-slip, flow driven by a meanVelocityForce
            fvConstraint (adaptive uniform momentum source imposing the bulk
            velocity; no entrance length). Re = u_mean * 2h / nu.
  couette — fixed wall at y = 0, wall moving at u_wall at y = H; no force, no
            pressure gradient — driven purely by the fixedValue BC.
            Re = u_wall * H / nu.

Each setup declares its extra template files (e.g. fvConstraints only exists
for channel), its Re -> nu mapping, its u_ref, and its extra provenance
(pressure_gradient only exists where a mean force does). The wall-normal
profile is sampled with the `sets` functionObject (`foamPostProcess -func
sample`); wall shear stress by the wallShearStress functionObject during the
solve.

All OpenFOAM dictionaries are rendered from string.Template files in
openfoam_templates/ — mesh resolution, viscosity and forcing are parameters,
never hand edits.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from string import Template

import numpy as np

TEMPLATE_DIR = Path(__file__).parent / "openfoam_templates"

# Templates use OpenFOAM 14 Foundation syntax (cellZone in fvConstraints,
# dictionary-form sets); pointing this at an older version will fail at
# dictionary parse. Override with BETAFLOW_OPENFOAM_BASHRC.
DEFAULT_BASHRC = "/opt/openfoam14/etc/bashrc"

# template file -> destination, shared by every case type
_SHARED_FILES = {
    "blockMeshDict": "system/blockMeshDict",
    "controlDict": "system/controlDict",
    "fvSchemes": "system/fvSchemes",
    "fvSolution": "system/fvSolution",
    "functions": "system/functions",
    "physicalProperties": "constant/physicalProperties",
    "momentumTransport": "constant/momentumTransport",
    "p": "0/p",
}

# case-type-specific templates on top of the shared set
_TYPE_FILES = {
    "channel": {"fvConstraints": "system/fvConstraints", "U_channel": "0/U"},
    "couette": {"U_couette": "0/U"},
}

# sampling mode -> template rendered into system/sample
_SAMPLINGS = {
    # Fixed uniform stations, cellPoint (linear) interpolation — how a user
    # typically extracts a profile; includes interpolation error.
    "cellPoint": "sample_cellPoint",
    # Raw cell-centre values via lineCell — the discrete solution where it is
    # defined; isolates the solver's error from profile extraction.
    "cell": "sample_cell",
}

_N_STREAMWISE = 4     # cyclic + x-invariant driving => solution is x-invariant
_N_SAMPLE_POINTS = 101
# Deep gate: the Couette null test requires the solve itself to sit at
# round-off, and unrelaxed SIMPLEC reaches this floor in tens of iterations.
_UX_RESIDUAL_TOL = 1e-12


def _end_time(n_cells):
    """SIMPLE iteration budget. The slowest error mode is diffusive, so
    iterations-to-round-off scale with N^2 (measured: the Couette residual
    reaches its 1e-15 floor at ~0.6 N^2 iterations; 0.8 N^2 leaves margin).
    Actual convergence is verified from the log, never assumed. The U linear
    solver's absolute tolerance must stay at round-off (1e-16): a looser
    value (1e-9) makes inner solves quit early and stalls the outer loop
    three decades above the floor."""
    return max(2000, (4 * int(n_cells) ** 2) // 5)


def _setup_channel(case, u_drive):
    """Force-driven periodic channel: walls at y = ±h, meanVelocityForce."""
    h = float(case["geometry"]["half_height"])
    # Re = u_mean * (2 h) / nu — bulk velocity, FULL channel height. Must
    # match betaflow/analytic/poiseuille.py; the test cross-checks meta.
    nu = u_drive * (2.0 * h) / float(case["nondim"]["Re"])
    u_ref = u_drive * float(case["normalisation"]["u_max_over_u_mean"])
    return {
        "y_min": -h,
        "y_max": h,
        "nu": nu,
        "u_ref": u_ref,
        "params": {"u_mean": u_drive, "u_init": u_drive},
        "meta": {"u_mean": u_drive},
    }


def _setup_couette(case, u_drive):
    """Wall-driven Couette: fixed wall at y = 0, moving wall at y = H.
    No force, no pressure gradient — so no pressure_gradient provenance."""
    height = float(case["geometry"]["height"])
    # Re = u_wall * H / nu — moving-wall speed, FULL gap height. Must match
    # betaflow/analytic/couette.py; the test cross-checks meta.
    nu = u_drive * height / float(case["nondim"]["Re"])
    return {
        "y_min": 0.0,
        "y_max": height,
        "nu": nu,
        "u_ref": u_drive,  # the oracle normalises by the moving-wall speed
        "params": {"u_wall": u_drive, "u_init": 0.5 * u_drive},
        "meta": {"u_wall": u_drive},
    }


_SETUPS = {"channel": _setup_channel, "couette": _setup_couette}


def run(case, **kwargs):
    """Dispatch on the case's geometry type.

    Steady types (channel, couette) share the residual-gated steady path;
    'womersley' uses the transient path whose convergence notion is
    cycle-to-cycle periodicity, not residual-to-steady. The two paths share
    meshing, template rendering, and execution machinery but differ in time
    controls, convergence verification, and post-processing.
    """
    if case["geometry"]["type"] == "womersley":
        return _run_womersley(case, **kwargs)
    return _run_steady(case, **kwargs)


def _run_steady(case, n_cells=40, u_drive=1.0, workdir=None, sampling="cellPoint"):
    """Run the case in OpenFOAM and return the standard profile dict.

    Parameters
    ----------
    case : dict
        Parsed YAML case definition (geometry, nondim, normalisation).
    n_cells : int
        Mesh level: number of cells across the full wall-normal gap.
    u_drive : float
        The case's driving velocity [m/s] — bulk velocity for 'channel',
        moving-wall speed for 'couette'. Sets the dimensional scale.
    workdir : path-like, optional
        Where to generate the OpenFOAM case (default: ./_runs). The case
        directory is kept for inspection and recreated from scratch each run.
    sampling : str
        "cellPoint" (default): lineUniform stations with cellPoint
        interpolation. "cell": raw cell-centre values, no interpolation.

    Returns
    -------
    dict
        {"y": ndarray [m], "u": ndarray [m/s], "u_ref": float [m/s],
         "tau_w": float [m^2/s^2], "meta": provenance dict}. u_ref is the
        velocity the case's oracle normalises by; tau_w is the kinematic
        wall-shear-stress magnitude from the wallShearStress functionObject.
        meta.pressure_gradient exists only for force-driven case types.
    """
    gtype = case["geometry"]["type"]
    if gtype not in _SETUPS:
        raise NotImplementedError(
            f"openfoam runner supports {sorted(_SETUPS)}, got '{gtype}'"
        )
    if sampling not in _SAMPLINGS:
        raise ValueError(f"unknown sampling '{sampling}': expected one of {sorted(_SAMPLINGS)}")

    setup = _SETUPS[gtype](case, u_drive)
    length = float(case["geometry"]["length"])
    gap = setup["y_max"] - setup["y_min"]

    workdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    casedir = workdir / f"{case['name']}_openfoam_n{int(n_cells)}_{sampling}"
    if casedir.exists():
        shutil.rmtree(casedir)

    eps = 5e-7 * gap  # keep sample endpoints strictly inside the mesh
    params = {
        "length": length,
        "y_min": setup["y_min"],
        "y_max": setup["y_max"],
        "thickness": 0.05 * gap,
        "nx": _N_STREAMWISE,
        "ny": int(n_cells),
        "nu": setup["nu"],
        "end_time": _end_time(n_cells),
        "x_mid": 0.5 * length,
        "z_mid": 0.025 * gap,
        "y_start": setup["y_min"] + eps,
        "y_end": setup["y_max"] - eps,
        "n_points": _N_SAMPLE_POINTS,
        **setup["params"],
    }
    _write_case(casedir, params, sampling, gtype)

    _foam(casedir, "blockMesh")
    _foam(casedir, "foamRun")
    _check_converged(casedir)
    _foam(casedir, "foamPostProcess -func sample -latestTime")

    y, u = _read_profile(casedir)
    tau_w = _read_tau_wall(casedir)

    meta = {
        "solver": "openfoam",
        "openfoam_version": _openfoam_version(),
        "mesh_level": int(n_cells),
        "n_cells_total": _N_STREAMWISE * int(n_cells),
        "nu": setup["nu"],
        "sampling": sampling,
        "case_dir": str(casedir),
        **setup["meta"],
    }
    if gtype == "channel":
        # Provenance of the driving force; only exists where a mean force does.
        meta["pressure_gradient"] = _read_pressure_gradient(casedir)

    return {"y": y, "u": u, "u_ref": setup["u_ref"], "tau_w": tau_w, "meta": meta}


def _write_case(casedir, params, sampling, gtype):
    files = {**_SHARED_FILES, **_TYPE_FILES[gtype]}
    files[_SAMPLINGS[sampling]] = "system/sample"
    for src, dest in files.items():
        target = casedir / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(Template((TEMPLATE_DIR / src).read_text()).substitute(params))


def _bashrc():
    return os.environ.get("BETAFLOW_OPENFOAM_BASHRC", DEFAULT_BASHRC)


def _foam(casedir, cmd):
    """Run one OpenFOAM command in the case directory, logging to log.<app>."""
    logfile = casedir / f"log.{cmd.split()[0]}"
    with open(logfile, "w") as f:
        proc = subprocess.run(
            ["bash", "-c", f"source {_bashrc()} 2>/dev/null && {cmd}"],
            cwd=casedir,
            stdout=f,
            stderr=subprocess.STDOUT,
        )
    if proc.returncode != 0:
        tail = "\n".join(logfile.read_text().splitlines()[-30:])
        raise RuntimeError(f"'{cmd}' failed in {casedir} (see {logfile}):\n{tail}")


def _check_converged(casedir):
    """Refuse to sample a run whose streamwise momentum has not converged.

    fvSolution deliberately has no residualControl: the wall-normal velocity
    and pressure are identically zero here, so their normalised residuals
    stagnate at O(0.1) regardless of convergence. The initial residual of the
    Ux equation is the meaningful indicator — it is normalised by the actual
    flow scale and drops to round-off when the solution is steady.
    """
    log = (casedir / "log.foamRun").read_text()
    residuals = re.findall(r"Solving for Ux, Initial residual = ([0-9eE.+-]+)", log)
    if not residuals:
        raise RuntimeError(f"no Ux residuals found in {casedir}/log.foamRun")
    final = float(residuals[-1])
    if final > _UX_RESIDUAL_TOL:
        raise RuntimeError(
            f"foamRun finished with Ux initial residual {final:.3e} > "
            f"{_UX_RESIDUAL_TOL:.0e} after {len(residuals)} iterations in "
            f"{casedir}; profile is not a converged steady solution. "
            f"See log.foamRun."
        )


def _openfoam_version():
    return subprocess.check_output(
        ["bash", "-c", f"source {_bashrc()} 2>/dev/null && echo -n $WM_PROJECT_VERSION"],
        text=True,
    ).strip()


def _read_pressure_gradient(casedir):
    """Converged mean pressure-gradient source [m/s^2] applied by the
    meanVelocityForce constraint, parsed from the last (corrected) print in
    the solver log. This is the G_disc of the discrete momentum balance."""
    log = (casedir / "log.foamRun").read_text()
    matches = re.findall(r"pressure gradient = ([0-9eE.+-]+)", log)
    if not matches:
        raise RuntimeError(f"no meanVelocityForce pressure gradient found in {casedir}/log.foamRun")
    return float(matches[-1])


def _read_tau_wall(casedir):
    """Mean kinematic wall-shear-stress magnitude over both wall patches.

    Parses the volVectorField the wallShearStress functionObject wrote at the
    final time. Every case type is x-invariant with equal-magnitude shear on
    the two walls (channel by symmetry, Couette by uniform shear), so all
    wall faces must carry the same magnitude — a larger spread means the
    solution is not fully developed and is reported rather than averaged away.
    """
    time_dir = max(
        (d for d in casedir.iterdir() if d.is_dir() and _is_time(d.name) and float(d.name) > 0),
        key=lambda d: float(d.name),
    )
    text = (time_dir / "wallShearStress").read_text()
    magnitudes = []
    for patch in ("bottomWall", "topWall"):
        block = re.search(patch + r"\s*\{([^}]*)\}", text, re.S)
        if block is None:
            raise RuntimeError(f"no '{patch}' patch in {time_dir}/wallShearStress")
        vectors = np.array(
            [
                [float(x) for x in triple.split()]
                for triple in re.findall(r"\(([^()]+)\)", block.group(1))
            ]
        )
        if vectors.size == 0:
            raise RuntimeError(f"no values parsed for '{patch}' in {time_dir}/wallShearStress")
        magnitudes.append(np.linalg.norm(vectors, axis=1))
    magnitudes = np.concatenate(magnitudes)
    spread = magnitudes.max() - magnitudes.min()
    # Allow iterative-convergence round-off (the Ux residual gate is 1e-9);
    # a genuinely undeveloped flow shows O(1) face-to-face variation.
    if spread > 1e-6 * magnitudes.mean():
        raise RuntimeError(
            f"wall shear stress varies across wall faces (spread {spread:.3e}); "
            f"flow is not fully developed in {casedir}"
        )
    return float(magnitudes.mean())


def _is_time(name):
    try:
        float(name)
        return True
    except ValueError:
        return False


def _read_profile(casedir):
    """Parse the raw sets output: columns y, U_x, U_y, U_z."""
    sample_root = casedir / "postProcessing" / "sample"
    times = sorted(
        (d for d in sample_root.iterdir() if d.is_dir()),
        key=lambda d: float(d.name),
    )
    data = np.loadtxt(times[-1] / "centreline.xy")
    return data[:, 0], data[:, 1]


# --------------------------------------------------------------------------
# Transient (womersley) path: convergence is cycle-to-cycle periodicity.
# --------------------------------------------------------------------------

_WOMERSLEY_FILES = {
    "blockMeshDict": "system/blockMeshDict",
    "controlDict_womersley": "system/controlDict",
    "fvSchemes_womersley": "system/fvSchemes",
    "fvSolution_womersley": "system/fvSolution",
    "functions_womersley": "system/functions",
    "fvModels_womersley": "constant/fvModels",
    "physicalProperties": "constant/physicalProperties",
    "momentumTransport": "constant/momentumTransport",
    "p": "0/p",
}

_N_PHASES = 16  # profile writes per cycle (periodicity check + harmonic fit)

_DDT_SCHEMES = {"backward": "backward", "Euler": "Euler"}


def _run_womersley(
    case,
    n_cells=None,
    alpha=None,
    n_steps=None,
    ddt="backward",
    g_amp=1.0,
    workdir=None,
    max_cycles=None,
):
    """Pulsatile plane channel driven by G cos(omega t); one continuous run.

    Runs max_cycles cycles in a single foamRun (cycle-restarts would
    re-bootstrap the backward scheme first-order at every cycle boundary and
    contaminate the temporal-order measurement); cycles_to_periodic is
    determined post-hoc from the logged cycle-to-cycle profile differences.

    The initial condition is the ANALYTIC solution at t = 0, written directly
    into 0/U at the cell centres: a quiescent start would need O(alpha^2)
    cycles for its transient to decay (slowest homogeneous-mode timescale
    (2/pi^3) alpha^2 T), so no accessible cap is meaningful without it.

    Returns
    -------
    dict
        {"y": stations [m], "u_amp"/"u_phase": first-harmonic amplitude [m/s]
        and phase [rad] at the stations, "u_ref": G/omega, "tau_amp"/
        "tau_phase": wall-shear first harmonic, "tau_ref": G*h, "meta": {...,
        cycles_to_periodic, periodicity (per-cycle ratios), identity_max_rel}}.
    """
    from betaflow.analytic import womersley as _wom

    if ddt not in _DDT_SCHEMES:
        raise ValueError(f"unknown ddt '{ddt}': expected one of {sorted(_DDT_SCHEMES)}")

    geom = case["geometry"]
    h = float(geom["half_height"])
    length = float(geom["length"])
    alpha = float(alpha if alpha is not None else case["nondim"]["alpha"])
    res = case["resolution"]
    n_steps = int(n_steps if n_steps is not None else res["n_steps_per_cycle"])
    conv = case["convergence"]
    max_cycles = int(max_cycles if max_cycles is not None else conv["max_cycles"])
    periodicity_tol = float(conv["periodicity_tol"])

    period = 1.0  # s; with power-of-two n_steps all write times are binary-exact
    omega = 2.0 * np.pi / period
    # alpha = h sqrt(omega/nu) => nu = omega h^2 / alpha^2 — must match
    # betaflow/analytic/womersley.py (half-height h).
    nu = omega * h**2 / alpha**2
    delta = np.sqrt(2.0 * nu / omega)
    if n_cells is None:
        # Mesh level = cells per Stokes layer, rounded to a multiple of 8.
        cpd = float(res["cells_per_stokes_layer"])
        n_cells = int(round(cpd * 2.0 * h / delta / 8.0)) * 8
    n_cells = int(n_cells)
    u_ref = g_amp / omega
    tau_ref = g_amp * h

    workdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    casedir = workdir / f"{case['name']}_openfoam_a{alpha:g}_n{n_cells}_nt{n_steps}_{ddt}"
    if casedir.exists():
        shutil.rmtree(casedir)

    gap = 2.0 * h
    eps = 5e-7 * gap
    params = {
        "length": length,
        "y_min": -h,
        "y_max": h,
        "thickness": 0.05 * gap,
        "nx": _N_STREAMWISE,
        "ny": n_cells,
        "nu": nu,
        "end_time": max_cycles * period,
        "delta_t": period / n_steps,
        "write_interval": period,
        "phase_interval": period / _N_PHASES,
        "ddt_scheme": _DDT_SCHEMES[ddt],
        "g_amp": g_amp,
        "frequency": 1.0 / period,
        "sine_start": -period / 4.0,  # sin(2 pi f (t + T/4)) = cos(omega t)
        "x_mid": 0.5 * length,
        "z_mid": 0.025 * gap,
        "y_start": -h + eps,
        "y_end": h - eps,
    }
    for src, dest in _WOMERSLEY_FILES.items():
        target = casedir / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(Template((TEMPLATE_DIR / src).read_text()).substitute(params))
    _write_womersley_ic(casedir, h, n_cells, alpha, u_ref, _wom)

    _foam(casedir, "blockMesh")
    _foam(casedir, "foamRun")

    profiles = _read_phase_profiles(casedir)
    ratios, cycles_to_periodic = _periodicity(profiles, period, max_cycles, periodicity_tol)
    y, u_hat = _harmonic_profile(profiles, period, max_cycles)

    t_series, tau_b, tau_t, u_bulk = _read_wall_series(casedir)
    identity_max, sign_b, sign_t, tau_series = _wall_identity(
        t_series, tau_b, tau_t, u_bulk, h, g_amp, omega, ddt, period / n_steps
    )
    tau_hat = _harmonic_scalar(t_series, tau_series, period, max_cycles)

    return {
        "y": y,
        "u_amp": np.abs(u_hat),
        "u_phase": np.angle(u_hat),
        "u_ref": u_ref,
        "tau_amp": float(np.abs(tau_hat)),
        "tau_phase": float(np.angle(tau_hat)),
        "tau_ref": tau_ref,
        "meta": {
            "solver": "openfoam",
            "openfoam_version": _openfoam_version(),
            "mesh_level": n_cells,
            "n_cells_total": _N_STREAMWISE * n_cells,
            "nu": nu,
            "alpha": alpha,
            "cells_per_stokes_layer": n_cells * delta / (2.0 * h),
            "n_steps_per_cycle": n_steps,
            "ddt": ddt,
            "cycles_run": max_cycles,
            "cycles_to_periodic": cycles_to_periodic,
            "periodicity": ratios,
            "identity_max_rel": identity_max,
            "wss_sign_convention": [sign_b, sign_t],
            "sampling": "cell",
            "case_dir": str(casedir),
        },
    }


def _write_womersley_ic(casedir, h, n_cells, alpha, u_ref, wom):
    """Write 0/U with the analytic t = 0 profile at the cell centres.

    blockMesh orders cells x-fastest, so each y-row's value repeats
    _N_STREAMWISE times consecutively.
    """
    dy = 2.0 * h / n_cells
    y_centres = -h + dy * (np.arange(n_cells) + 0.5)
    u0 = u_ref * np.real(wom.complex_profile(y_centres / h, alpha))
    values = np.repeat(u0, _N_STREAMWISE)
    entries = "\n".join(f"({v:.12g} 0 0)" for v in values)
    (casedir / "0" / "U").write_text(
        "FoamFile\n{\n    format      ascii;\n    class       volVectorField;\n"
        "    object      U;\n}\n\n"
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        f"internalField   nonuniform List<vector>\n{values.size}\n(\n{entries}\n);\n\n"
        "boundaryField\n{\n"
        "    inlet        { type cyclic; }\n"
        "    outlet       { type cyclic; }\n"
        "    bottomWall   { type noSlip; }\n"
        "    topWall      { type noSlip; }\n"
        "    frontAndBack { type empty; }\n"
        "}\n"
    )


def _read_phase_profiles(casedir):
    """All sets-FO profile writes: {time: (y, ux)}, sorted by time."""
    root = casedir / "postProcessing" / "profile"
    out = {}
    for d in root.iterdir():
        if d.is_dir():
            data = np.loadtxt(d / "centreline.xy")
            out[float(d.name)] = (data[:, 0], data[:, 1])
    return dict(sorted(out.items()))


def _profile_at(profiles, t):
    for key, value in profiles.items():
        if abs(key - t) < 1e-9:
            return value
    raise RuntimeError(f"no sampled profile at t = {t}")


def _periodicity(profiles, period, max_cycles, tol):
    """Cycle-to-cycle periodicity ratios and the first cycle meeting tol.

    r_k = max over the cycle's phases of ||u_k(t) - u_{k-1}(t - T)||_2,
    normalised by the cycle's maximum profile norm (the literal ||u(t)||
    denominator vanishes at flow reversal, so the cycle max is used).
    """
    ratios = []
    cycles_to_periodic = None
    for k in range(2, max_cycles + 1):
        diffs, norms = [], []
        for m in range(1, _N_PHASES + 1):
            t2 = (k - 1) * period + m * period / _N_PHASES
            _, u2 = _profile_at(profiles, t2)
            _, u1 = _profile_at(profiles, t2 - period)
            diffs.append(np.linalg.norm(u2 - u1))
            norms.append(np.linalg.norm(u2))
        ratio = max(diffs) / max(norms)
        ratios.append(float(ratio))
        if cycles_to_periodic is None and ratio < tol:
            cycles_to_periodic = k
    return ratios, cycles_to_periodic


def _harmonic_profile(profiles, period, max_cycles):
    """First-harmonic complex profile from the final cycle's phase samples.

    The discrete problem is linear and time-invariant under single-harmonic
    forcing, so once periodic the solution is single-harmonic and the
    _N_PHASES-point DFT bin is exact (no leakage from truncation).
    """
    acc = None
    y = None
    for m in range(1, _N_PHASES + 1):
        t = (max_cycles - 1) * period + m * period / _N_PHASES
        y, u = _profile_at(profiles, t)
        term = u * np.exp(-2.0j * np.pi * t / period)
        acc = term if acc is None else acc + term
    return y, 2.0 / _N_PHASES * acc


def _harmonic_scalar(t, series, period, max_cycles):
    """First-harmonic complex amplitude of a per-timestep scalar series,
    evaluated over the final cycle."""
    mask = t > (max_cycles - 1) * period + 1e-9
    tt, ss = t[mask], series[mask]
    return 2.0 / tt.size * np.sum(ss * np.exp(-2.0j * np.pi * tt / period))


def _read_vector_series(path):
    times, vectors = [], []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"([0-9.eE+-]+)\s+\(([^)]*)\)", line)
        times.append(float(match.group(1)))
        vectors.append([float(x) for x in match.group(2).split()])
    return np.array(times), np.array(vectors)


def _read_wall_series(casedir):
    """Per-timestep x-components: wall shear on each wall and bulk velocity."""
    root = casedir / "postProcessing"
    t_b, tau_b = _read_vector_series(root / "tauBottom" / "0" / "surfaceFieldValue.dat")
    t_t, tau_t = _read_vector_series(root / "tauTop" / "0" / "surfaceFieldValue.dat")
    t_u, u_bulk = _read_vector_series(root / "uBulk" / "0" / "volFieldValue.dat")
    if not (t_b.size == t_t.size == t_u.size):
        raise RuntimeError(f"wall/bulk series lengths differ in {casedir}")
    return t_b, tau_b[:, 0], tau_t[:, 0], u_bulk[:, 0]


def _wall_identity(t, tau_bx, tau_tx, ub_x, h, g_amp, omega, ddt, delta_t):
    """Per-timestep momentum-balance identity tau_w(t) = h (G(t) - d<u>/dt).

    d<u>/dt uses the SAME discrete operator as the solver's ddt scheme
    (backward: (3u^n - 4u^{n-1} + u^{n-2}) / 2dt), skipping the bootstrap
    steps. G(t^n) = g cos(omega t^n) exactly — semiImplicitSource evaluates
    its Function1 at the current time. The wallShearStress sign convention
    (R.n with n into the domain, per wall) is resolved empirically over the
    four sign combinations and reported; the flow is symmetric so both walls
    must agree.
    """
    g = g_amp * np.cos(omega * t)
    dudt = np.full_like(ub_x, np.nan)
    if ddt == "backward":
        dudt[2:] = (3.0 * ub_x[2:] - 4.0 * ub_x[1:-1] + ub_x[:-2]) / (2.0 * delta_t)
    else:
        dudt[1:] = (ub_x[1:] - ub_x[:-1]) / delta_t
    balance = h * (g - dudt)
    valid = ~np.isnan(dudt)
    best = None
    for sign_b in (1, -1):
        for sign_t in (1, -1):
            tau = 0.5 * (sign_b * tau_bx + sign_t * tau_tx)
            scale = np.max(np.abs(tau[valid]))
            residual = np.max(np.abs(tau[valid] - balance[valid])) / scale
            if best is None or residual < best[0]:
                best = (float(residual), sign_b, sign_t, tau)
    return best
