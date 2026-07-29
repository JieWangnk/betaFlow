"""OpenFOAM 12 (Foundation) adapter — the only module that knows OpenFOAM exists.

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

DEFAULT_BASHRC = "/opt/openfoam12/etc/bashrc"

# template file -> destination inside the generated case directory
_FILES = {
    "blockMeshDict": "system/blockMeshDict",
    "controlDict": "system/controlDict",
    "fvSchemes": "system/fvSchemes",
    "fvSolution": "system/fvSolution",
    "fvConstraints": "system/fvConstraints",
    "sample": "system/sample",
    "physicalProperties": "constant/physicalProperties",
    "momentumTransport": "constant/momentumTransport",
    "U": "0/U",
    "p": "0/p",
}

_N_STREAMWISE = 4     # cyclic + uniform forcing => solution is x-invariant
_N_SAMPLE_POINTS = 101
_UX_RESIDUAL_TOL = 1e-9


def _end_time(n_cells):
    """SIMPLE iteration budget. Convergence slows with refinement (429 / 651 /
    >1000 iterations to reach a 1e-9 Ux residual at N = 40/80/160), so scale
    the cap with mesh level; actual convergence is verified from the log."""
    return max(1000, 50 * int(n_cells))


def run(case, n_cells=40, u_mean=1.0, workdir=None):
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

    Returns
    -------
    dict
        {"y": ndarray [m], "u": ndarray [m/s], "u_ref": float [m/s],
         "meta": provenance dict}. u_ref is the analytic peak velocity implied
        by the imposed bulk velocity and the case's normalisation entry.
    """
    geom = case["geometry"]
    if geom["type"] != "channel":
        raise NotImplementedError(f"openfoam runner only supports 'channel', got '{geom['type']}'")

    h = float(geom["half_height"])
    length = float(geom["length"])
    reynolds = float(case["nondim"]["Re"])

    # Re = u_mean * (2 h) / nu — bulk velocity, FULL channel height. This must
    # match the definition in betaflow/analytic/poiseuille.py; the test
    # cross-checks (u_mean, nu) reported in meta against that definition.
    nu = u_mean * (2.0 * h) / reynolds

    u_ref = u_mean * float(case["normalisation"]["u_max_over_u_mean"])

    workdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    casedir = workdir / f"{case['name']}_openfoam_n{int(n_cells)}"
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
    _write_case(casedir, params)

    _foam(casedir, "blockMesh")
    _foam(casedir, "foamRun")
    _check_converged(casedir)
    _foam(casedir, "foamPostProcess -func sample -latestTime")

    y, u = _read_profile(casedir)

    return {
        "y": y,
        "u": u,
        "u_ref": u_ref,
        "meta": {
            "solver": "openfoam",
            "openfoam_version": _openfoam_version(),
            "mesh_level": int(n_cells),
            "n_cells_total": _N_STREAMWISE * int(n_cells),
            "nu": nu,
            "u_mean": u_mean,
            "case_dir": str(casedir),
        },
    }


def _write_case(casedir, params):
    for src, dest in _FILES.items():
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


def _read_profile(casedir):
    """Parse the raw sets output: columns y, U_x, U_y, U_z."""
    sample_root = casedir / "postProcessing" / "sample"
    times = sorted(
        (d for d in sample_root.iterdir() if d.is_dir()),
        key=lambda d: float(d.name),
    )
    data = np.loadtxt(times[-1] / "centreline.xy")
    return data[:, 0], data[:, 1]
