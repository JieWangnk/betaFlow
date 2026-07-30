"""OpenFOAM 14 (Foundation) adapter — the only module that knows OpenFOAM exists.

Channel cases are meshed one cell thick in z with `empty` front/back patches
and cyclic streamwise patches. The flow is driven by a `meanVelocityForce`
fvConstraint (an adaptive uniform momentum source), which imposes the bulk
velocity directly and eliminates entrance-length effects, so the domain can be
short. The wall-normal profile is sampled with the `sets` functionObject
(`foamPostProcess -func sample`) on the converged fields.

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

# template file -> destination inside the generated case directory
_FILES = {
    "blockMeshDict": "system/blockMeshDict",
    "controlDict": "system/controlDict",
    "fvSchemes": "system/fvSchemes",
    "fvSolution": "system/fvSolution",
    "fvConstraints": "system/fvConstraints",
    "functions": "system/functions",
    "physicalProperties": "constant/physicalProperties",
    "momentumTransport": "constant/momentumTransport",
    "U": "0/U",
    "p": "0/p",
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

_N_STREAMWISE = 4     # cyclic + uniform forcing => solution is x-invariant
_N_SAMPLE_POINTS = 101
_UX_RESIDUAL_TOL = 1e-9


def _end_time(n_cells):
    """SIMPLE iteration budget. Convergence slows with refinement (429 / 651 /
    >1000 iterations to reach a 1e-9 Ux residual at N = 40/80/160), so scale
    the cap with mesh level; actual convergence is verified from the log."""
    return max(1000, 50 * int(n_cells))


def run(case, n_cells=40, u_mean=1.0, workdir=None, sampling="cellPoint"):
    """Run the case in OpenFOAM and return the standard profile dict.

    Parameters
    ----------
    case : dict
        Parsed YAML case definition (geometry, nondim, normalisation).
    n_cells : int
        Mesh level: number of cells across the FULL channel height 2h.
    u_mean : float
        Imposed bulk velocity [m/s]; sets the dimensional scale of the run.
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
        analytic peak velocity implied by the imposed bulk velocity and the
        case's normalisation entry; tau_w is the kinematic wall-shear-stress
        magnitude from the wallShearStress functionObject.
    """
    geom = case["geometry"]
    if geom["type"] != "channel":
        raise NotImplementedError(f"openfoam runner only supports 'channel', got '{geom['type']}'")
    if sampling not in _SAMPLINGS:
        raise ValueError(f"unknown sampling '{sampling}': expected one of {sorted(_SAMPLINGS)}")

    h = float(geom["half_height"])
    length = float(geom["length"])
    reynolds = float(case["nondim"]["Re"])

    # Re = u_mean * (2 h) / nu — bulk velocity, FULL channel height. This must
    # match the definition in betaflow/analytic/poiseuille.py; the test
    # cross-checks (u_mean, nu) reported in meta against that definition.
    nu = u_mean * (2.0 * h) / reynolds

    u_ref = u_mean * float(case["normalisation"]["u_max_over_u_mean"])

    workdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    casedir = workdir / f"{case['name']}_openfoam_n{int(n_cells)}_{sampling}"
    if casedir.exists():
        shutil.rmtree(casedir)

    eps = 1e-6 * h  # keep sample endpoints strictly inside the mesh
    params = {
        "length": length,
        "half_height": h,
        "neg_half_height": -h,
        "thickness": 0.1 * h,
        "nx": _N_STREAMWISE,
        "ny": int(n_cells),
        "nu": nu,
        "u_mean": u_mean,
        "end_time": _end_time(n_cells),
        "x_mid": 0.5 * length,
        "z_mid": 0.05 * h,
        "y_start": -(h - eps),
        "y_end": h - eps,
        "n_points": _N_SAMPLE_POINTS,
    }
    _write_case(casedir, params, sampling)

    _foam(casedir, "blockMesh")
    _foam(casedir, "foamRun")
    _check_converged(casedir)
    _foam(casedir, "foamPostProcess -func sample -latestTime")

    y, u = _read_profile(casedir)
    tau_w = _read_tau_wall(casedir)
    g_disc = _read_pressure_gradient(casedir)

    return {
        "y": y,
        "u": u,
        "u_ref": u_ref,
        "tau_w": tau_w,
        "meta": {
            "solver": "openfoam",
            "openfoam_version": _openfoam_version(),
            "mesh_level": int(n_cells),
            "n_cells_total": _N_STREAMWISE * int(n_cells),
            "nu": nu,
            "u_mean": u_mean,
            "pressure_gradient": g_disc,
            "sampling": sampling,
            "case_dir": str(casedir),
        },
    }


def _write_case(casedir, params, sampling):
    files = dict(_FILES)
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
    """Mean kinematic wall-shear-stress magnitude over the wall patch faces.

    Parses the volVectorField the wallShearStress functionObject wrote at the
    final time. The flow is x-invariant, so every wall face must carry the
    same magnitude — a spread larger than round-off means the solution is not
    fully developed and is reported as an error rather than averaged away.
    """
    time_dir = max(
        (d for d in casedir.iterdir() if d.is_dir() and _is_time(d.name) and float(d.name) > 0),
        key=lambda d: float(d.name),
    )
    text = (time_dir / "wallShearStress").read_text()
    walls = re.search(r"walls\s*\{([^}]*)\}", text, re.S)
    if walls is None:
        raise RuntimeError(f"no 'walls' patch in {time_dir}/wallShearStress")
    vectors = np.array(
        [[float(x) for x in triple.split()] for triple in re.findall(r"\(([^()]+)\)", walls.group(1))]
    )
    if vectors.size == 0:
        raise RuntimeError(f"no wall values parsed from {time_dir}/wallShearStress")
    magnitudes = np.linalg.norm(vectors, axis=1)
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
