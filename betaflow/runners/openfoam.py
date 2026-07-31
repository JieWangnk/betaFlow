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
# Genuinely geometry-independent files only. blockMeshDict and 0/p were in
# here until the pipe port: both encode the box topology and its patch names,
# so both are now declared per case type. Overriding a "shared" entry by
# dict-merge order is not a contract (see the momentumTransport note below).
_SHARED_FILES = {
    "controlDict": "system/controlDict",
    "fvSchemes": "system/fvSchemes",
    "fvSolution": "system/fvSolution",
    "functions": "system/functions",
    "physicalProperties": "constant/physicalProperties",
}

_BOX_FILES = {"blockMeshDict": "system/blockMeshDict", "p": "0/p"}
_WEDGE_FILES = {"blockMeshDict_wedge": "system/blockMeshDict", "p_pipe": "0/p"}

# Case-type-specific templates on top of the shared set. Each type names its
# own constant/momentumTransport: adding a constitutive model must not mean
# silently overriding a shared entry (dict-merge order is not a contract).
_TYPE_FILES = {
    "channel": {
        **_BOX_FILES,
        "fvConstraints": "system/fvConstraints",
        "U_channel": "0/U",
        "momentumTransport": "constant/momentumTransport",
    },
    "couette": {
        **_BOX_FILES,
        "U_couette": "0/U",
        "momentumTransport": "constant/momentumTransport",
    },
    "casson": {
        **_BOX_FILES,
        "fvConstraints": "system/fvConstraints",
        "U_channel": "0/U",
        "momentumTransport_casson": "constant/momentumTransport",
        "functions_casson": "system/functions",
    },
    "carreau": {
        **_BOX_FILES,
        "fvConstraints": "system/fvConstraints",
        "U_channel": "0/U",
        "momentumTransport_carreau": "constant/momentumTransport",
    },
    # --- circular pipe, axisymmetric wedge --------------------------------
    "pipe": {
        **_WEDGE_FILES,
        "fvConstraints": "system/fvConstraints",
        "U_pipe": "0/U",
        "momentumTransport": "constant/momentumTransport",
    },
    "pipe_casson": {
        **_WEDGE_FILES,
        "fvConstraints": "system/fvConstraints",
        "U_pipe": "0/U",
        "momentumTransport_casson": "constant/momentumTransport",
        "functions_casson": "system/functions",
    },
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

# HARDCODING FOUND (pipe port): the tau parser assumed the channel's
# bottomWall/topWall pair. A pipe has ONE wall, so the patch list is now a
# property of the case type rather than a constant in the parser.
_WALL_PATCHES = {
    "channel": ("bottomWall", "topWall"),
    "couette": ("bottomWall", "topWall"),
    "casson": ("bottomWall", "topWall"),
    "carreau": ("bottomWall", "topWall"),
    "womersley": ("bottomWall", "topWall"),
    "womersley_carreau": ("bottomWall", "topWall"),
    "pipe": ("wall",),
    "pipe_casson": ("wall",),
    "pipe_womersley": ("wall",),
}

_N_STREAMWISE = 4     # cyclic + x-invariant driving => solution is x-invariant
_N_SAMPLE_POINTS = 101
# Deep gate: the Couette null test requires the solve itself to sit at
# round-off, and unrelaxed SIMPLEC reaches this floor in tens of iterations.
_UX_RESIDUAL_TOL = 1e-12

# Casson's nonlinear viscosity fixed point converges algebraically, not
# geometrically: at nuMax/nu_c = 1e5 the Ux residual is still ~1e-6 after
# 60k iterations, and this is NOT a linear-solver artefact (PBiCGStab/DILU
# and smoothSolver stall at bit-identical values). The measured plug width
# and profile stabilise far earlier — by residual ~1e-5 — so these runs are
# given a fixed budget and gated on measurement drift instead.
_CASSON_ITERATIONS = 20000


# Carreau's nu<->gammadot fixed point contracts GEOMETRICALLY (measured rate
# 0.9948/iteration at N=40 — a constant ratio, unlike casson's algebraic
# decay). Cost is NOT independent of Cu, as was predicted: measured
# iterations-to-1e-10 at N=80 are 755 / 898 / 3916 / 15344 for Cu =
# 0.1 / 1 / 10 / 100, i.e. ~900*Cu^0.6 once Cu > 1, against 764 for the
# Newtonian channel on the same mesh. At Cu -> 0 it converges to the
# Newtonian count exactly; the growth tracks the viscosity contrast the
# thinning creates across the channel, which Carreau BOUNDS at nu0/nuInf
# whereas casson's nuMax makes it unbounded.
def _carreau_iterations(cu, n_cells):
    return int(max(6000.0, 3000.0 * cu**0.6) * max(1.0, n_cells / 80.0))


def _end_time(n_cells, gtype="channel"):
    """SIMPLE iteration budget. The slowest error mode is diffusive, so
    iterations-to-round-off scale with N^2 (measured: the Couette residual
    reaches its 1e-15 floor at ~0.6 N^2 iterations; 0.8 N^2 leaves margin).
    Actual convergence is verified from the log, never assumed. The U linear
    solver's absolute tolerance must stay at round-off (1e-16): a looser
    value (1e-9) makes inner solves quit early and stalls the outer loop
    three decades above the floor."""
    return max(2000, (4 * int(n_cells) ** 2) // 5)


_WEDGE_HALF_ANGLE_DEG = 2.5  # 5 degrees total; OpenFOAM requires < 5 per side


def _wedge_params(a, n_cells):
    """Geometry parameters for the axisymmetric wedge block.

    FACETING CORRECTION. A wedge's outer boundary is a flat chord, not an arc.
    Putting the vertices on the circle r = a gives a discrete volume-to-area
    ratio of a*cos(theta)/2, so the exact discrete force balance yields
    tau_w = G a cos(theta)/2 rather than G a / 2 — a relative bias of
    1 - cos(theta) = 9.52e-4 at theta = 2.5 deg that is INDEPENDENT of radial
    refinement and therefore invisible in a mesh-convergence study. Measured
    at exactly that value before this correction.

    Placing the vertices at a/cos(theta) instead puts the chord MIDPOINT on
    the circle of radius a, restoring V/A = a/2 identically.
    """
    th = np.radians(_WEDGE_HALF_ANGLE_DEG)
    r_v = a / np.cos(th)
    return {
        "r_cos": r_v * np.cos(th),
        "r_sin": r_v * np.sin(th),
        "r_nsin": -r_v * np.sin(th),
        "nr": int(n_cells),
        # The sample line must sit on the wedge symmetry plane (z = 0), not
        # at a fraction of the domain thickness as the box cases use.
        "z_mid": 0.0,
    }


def _setup_pipe(case, u_drive, n_cells=None):
    """Circular pipe, Newtonian, force-driven — the pipe twin of 'channel'.

    Re = u_mean * (2a) / nu uses the pipe DIAMETER (see
    betaflow/analytic/pipe.py for the citation); u_max/u_mean = 2, not the
    channel's 3/2.
    """
    from betaflow.analytic import pipe as _pipe

    a = float(case["geometry"]["radius"])
    nu = u_drive * (2.0 * a) / float(case["nondim"]["Re"])
    return {
        "y_min": 0.0,          # the axis, r = 0
        "y_max": a,
        "nu": nu,
        "u_ref": u_drive * _pipe.POISEUILLE_U_MAX_OVER_U_MEAN,
        "params": {"u_mean": u_drive, "u_init": u_drive,
                   **_wedge_params(a, n_cells)},
        "meta": {"u_mean": u_drive, "radius": a},
    }


def _setup_pipe_casson(case, u_drive, n_cells=None):
    """Circular pipe, Casson. Plug radius r_p = 2 tau_y / G (factor 2 vs the
    channel's tau_y/G), reported as xi_c = r_p/a."""
    from betaflow.analytic import pipe as _pipe

    a = float(case["geometry"]["radius"])
    xi_c = float(case["nondim"]["xi_c"])
    nu_c = u_drive * (2.0 * a) / float(case["nondim"]["Re"])
    g_pred = _pipe.casson_pressure_gradient_for_bulk(u_drive, a, nu_c, xi_c)
    tau0 = xi_c * g_pred * a / 2.0
    nu_max = float(case["regularisation"]["nu_max_ratio"]) * nu_c
    u_ref = float(_pipe.casson_velocity(0.0, g_pred, tau0, nu_c, a))
    return {
        "y_min": 0.0,
        "y_max": a,
        "nu": nu_c,
        "u_ref": u_ref,
        "params": {
            "u_mean": u_drive, "u_init": u_drive,
            "nu_c": nu_c, "tau0": tau0, "nu_max": nu_max,
            "fields": "U strainRateViscosityModel:nu",
            **_wedge_params(a, n_cells),
        },
        "meta": {
            "u_mean": u_drive, "radius": a, "xi_c": xi_c, "nu_c": nu_c,
            "tau0": tau0, "nu_max": nu_max, "g_predicted": g_pred,
        },
    }


def _setup_carreau(case, u_drive, cu=None, fluid=None, nu_inf_over_nu0=None):
    """Carreau-Yasuda channel: same drive and geometry as 'channel'.

    Two parameterisations. With `cu`, the Carreau number is TARGETED: since
    Cu = k G h / nu0 couples k and G (each depends on the other), the oracle
    solves both jointly for the requested bulk velocity. With `fluid`, a named
    real fluid from the case file fixes k in seconds and Cu is an OUTPUT.
    """
    from betaflow.analytic import carreau as _carreau

    nd = case["nondim"]
    a = float(nd.get("a", 2.0))
    if fluid is not None:
        spec = case["fluids"][fluid]
        rho = float(spec["rho"])
        nu0 = float(spec["mu_0"]) / rho
        nu_inf = float(spec["mu_inf"]) / rho
        k = float(spec["lambda"])
        n = float(spec["n"])
        h = float(spec["half_height"])
        u_drive = float(spec["u_mean"])
        length = float(spec["length"])
        g_pred = _carreau.pressure_gradient_for_bulk(u_drive, h, nu0, nu_inf, k, n, a)
    else:
        h = float(case["geometry"]["half_height"])
        length = float(case["geometry"]["length"])
        n = float(nd["n"])
        # Re uses nu0, the ZERO-shear viscosity. Must match
        # betaflow/analytic/carreau.py; the test cross-checks meta.
        nu0 = u_drive * (2.0 * h) / float(nd["Re"])
        ratio = nd["nu_inf_over_nu0"] if nu_inf_over_nu0 is None else nu_inf_over_nu0
        nu_inf = float(ratio) * nu0
        cu = float(nd["Cu"] if cu is None else cu)
        g_pred, k = _carreau.drive_for_carreau_number(u_drive, h, nu0, nu_inf, n, cu, a)

    u_ref = _carreau.u_max(g_pred, h, nu0, nu_inf, k, n, a)
    return {
        "y_min": -h,
        "y_max": h,
        "length": length,
        "nu": nu0,  # physicalProperties nu IS the zero-shear viscosity here
        "u_ref": u_ref,
        "u_drive": u_drive,
        "params": {
            "u_mean": u_drive,
            "u_init": u_drive,
            "nu_inf": nu_inf,
            "k": k,
            "n": n,
            "a": a,
        },
        "meta": {
            "u_mean": u_drive,
            "nu0": nu0,
            "nu_inf": nu_inf,
            "k": k,
            "n": n,
            "a": a,
            "half_height": h,
            "fluid": fluid,
            "carreau_number_predicted": _carreau.carreau_number(k, g_pred, h, nu0),
            "g_predicted": g_pred,
        },
    }


def _setup_casson(case, u_drive, nu_max_ratio=1.0e4):
    """Casson channel: identical drive and geometry to 'channel'; what differs
    is the constitutive model and its regularisation cap.

    The yield stress is set so the case realises its target xi = tau_y/(G h)
    at the analytic G for this bulk velocity. G_disc read back from the solver
    is what the plug width is actually computed from — never the prescribed G.
    """
    from betaflow.analytic import casson as _casson

    h = float(case["geometry"]["half_height"])
    xi = float(case["nondim"]["xi"])
    # Re uses nu_c, the Casson consistency (infinite-shear limit), matching
    # the poiseuille convention. Must match betaflow/analytic/casson.py.
    nu_c = u_drive * (2.0 * h) / float(case["nondim"]["Re"])
    g_pred = _casson.pressure_gradient_for_bulk(u_drive, h, nu_c, xi)
    tau0 = xi * g_pred * h
    nu_max = nu_max_ratio * nu_c
    u_ref = g_pred * h**2 / nu_c * _casson.u_max_over_U0(xi)
    return {
        "y_min": -h,
        "y_max": h,
        "nu": nu_c,
        "u_ref": u_ref,
        "params": {
            "u_mean": u_drive,
            "u_init": u_drive,
            "nu_c": nu_c,
            "tau0": tau0,
            "nu_max": nu_max,
        },
        "meta": {
            "u_mean": u_drive,
            "xi": xi,
            "nu_c": nu_c,
            "tau0": tau0,
            "nu_max": nu_max,
            "nu_max_ratio": nu_max_ratio,
            "g_predicted": g_pred,
        },
    }


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


_SETUPS = {
    "pipe": _setup_pipe,
    "pipe_casson": _setup_pipe_casson,
    "channel": _setup_channel,
    "couette": _setup_couette,
    "casson": _setup_casson,
    "carreau": _setup_carreau,
}

# Case types whose drive is a mean force, so meta carries pressure_gradient.
_FORCE_DRIVEN = {"channel", "casson", "carreau", "pipe", "pipe_casson"}

# Generalised-Newtonian case types are gated on the CONSERVATION IDENTITY
# tau_w = G_disc*h rather than on the solver residual. Residual and profile
# drift measure step size; a nonlinear nu<->gammadot fixed point that
# contracts slowly looks converged by both long before it is (measured in
# casson: residual 2.2e-9 and drift 9.7e-6 with the identity still at 2.5e-3).
# The identity compares two quantities that must agree AT the fixed point, so
# it measures distance to the solution. It is also rheology-independent, so
# it cannot be satisfied by a wrong constitutive model.
_GENERALISED_NEWTONIAN = {"casson", "carreau", "pipe_casson"}
_IDENTITY_GATE_TOL = 1e-6


def run(case, **kwargs):
    """Dispatch on the case's geometry type.

    Steady types (channel, couette) share the residual-gated steady path;
    'womersley' uses the transient path whose convergence notion is
    cycle-to-cycle periodicity, not residual-to-steady. The two paths share
    meshing, template rendering, and execution machinery but differ in time
    controls, convergence verification, and post-processing.
    """
    if case["geometry"]["type"] in ("womersley", "womersley_carreau", "pipe_womersley"):
        return _run_womersley(case, **kwargs)
    return _run_steady(case, **kwargs)


def _run_steady(
    case,
    n_cells=40,
    u_drive=1.0,
    workdir=None,
    sampling="cellPoint",
    require_converged=True,
    iterations=None,
    **setup_kwargs,
):
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

    if gtype.startswith("pipe"):
        setup_kwargs = {**setup_kwargs, "n_cells": int(n_cells)}
    setup = _SETUPS[gtype](case, u_drive, **setup_kwargs)
    length = float(setup.get("length", case["geometry"]["length"]))
    u_drive = setup.get("u_drive", u_drive)
    gap = setup["y_max"] - setup["y_min"]

    workdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    suffix = "".join(
        f"_{k}{v:g}" if isinstance(v, (int, float)) else f"_{k}{v}"
        for k, v in sorted(setup_kwargs.items())
        if v is not None
    )
    casedir = workdir / f"{case['name']}_openfoam_n{int(n_cells)}{suffix}_{sampling}"
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
        "end_time": (
            int(iterations or _CASSON_ITERATIONS)
            if gtype in ("casson", "pipe_casson")
            else _carreau_iterations(setup["meta"]["carreau_number_predicted"], n_cells)
            if gtype == "carreau"
            else _end_time(n_cells, gtype)
        ),
        # Casson writes a second, half-way state: its nonlinear fixed point
        # converges too slowly for a residual gate, so convergence is judged
        # on the drift of the measured profile between the two states.
        "write_interval": (
            int(iterations or _CASSON_ITERATIONS) // 2
            if gtype in ("casson", "pipe_casson")
            else _carreau_iterations(setup["meta"]["carreau_number_predicted"], n_cells)
            if gtype == "carreau"
            else _end_time(n_cells, gtype)
        ),
        "x_mid": 0.5 * length,
        "z_mid": 0.025 * gap,
        "y_start": setup["y_min"] + eps,
        "y_end": setup["y_max"] - eps,
        "n_points": _N_SAMPLE_POINTS,
        # Casson also samples the effective viscosity: the cap-active region
        # is exactly where it equals nuMax. The generalisedNewtonian model
        # registers it under this name (not "nuEff", which is not in the
        # registry).
        "fields": "U strainRateViscosityModel:nu" if gtype == "casson" else "U",
        **setup["params"],
    }
    _write_case(casedir, params, sampling, gtype)

    _foam(casedir, "blockMesh")
    if gtype in ("casson",):
        # Start from the analytic Casson profile. The nonlinear viscosity
        # fixed point converges algebraically from a uniform start, so a
        # good initial nu field is worth orders of magnitude in iterations;
        # only the regularisation correction then has to develop.
        _write_channel_ic(casedir, setup, params, case, n_cells)
    _foam(casedir, "foamRun")
    residual = _final_ux_residual(casedir)
    latest = "" if gtype in ("casson", "pipe_casson") else " -latestTime"
    _foam(casedir, f"foamPostProcess -func sample{latest}")

    y, u = _read_profile(casedir)
    tau_w = _read_tau_wall(casedir, _WALL_PATCHES[gtype])

    meta = {
        "solver": "openfoam",
        "openfoam_version": _openfoam_version(),
        "mesh_level": int(n_cells),
        "n_cells_total": _N_STREAMWISE * int(n_cells),
        "nu": setup["nu"],
        "sampling": sampling,
        "ux_residual": residual,
        "iterations_run": _iteration_count(casedir),
        "iterations_to_residual_1e10": _iterations_to_residual(casedir, 1e-10),
        "case_dir": str(casedir),
        **setup["meta"],
    }
    if gtype in _FORCE_DRIVEN:
        # Provenance of the driving force; only exists where a mean force does.
        meta["pressure_gradient"] = _read_pressure_gradient(casedir)

    # --- convergence gate ---------------------------------------------------
    # Generalised-Newtonian cases are gated on the conservation identity;
    # Newtonian ones on the residual (their fixed point is linear, so the
    # residual is a faithful proxy and reaches round-off). See
    # _GENERALISED_NEWTONIAN for why the distinction matters.
    if gtype in _GENERALISED_NEWTONIAN:
        h_gate = 0.5 * (setup["y_max"] - setup["y_min"])
        balance = meta["pressure_gradient"] * h_gate
        meta["identity"] = abs(tau_w - balance) / abs(balance)
        if require_converged and meta["identity"] > _IDENTITY_GATE_TOL:
            raise RuntimeError(
                f"conservation identity tau_w = G_disc*h open by "
                f"{meta['identity']:.3e} > {_IDENTITY_GATE_TOL:.0e} in "
                f"{casedir}; the nonlinear fixed point has not converged "
                f"(Ux residual {residual:.2e} is a step-size measure and does "
                f"NOT establish this). See log.foamRun."
            )
    elif require_converged and residual > _UX_RESIDUAL_TOL:
        raise RuntimeError(
            f"foamRun finished with Ux initial residual {residual:.3e} > "
            f"{_UX_RESIDUAL_TOL:.0e} in {casedir}; profile is not a converged "
            f"steady solution. See log.foamRun."
        )

    result = {"y": y, "u": u, "u_ref": setup["u_ref"], "tau_w": tau_w, "meta": meta}
    if gtype in ("casson", "pipe_casson"):
        # Effective viscosity at the sampled cell centres, as the solver
        # computed it — needed to locate the cap-active (nu == nuMax) region.
        result["nu_eff"] = _read_nu_eff(casedir)
        meta["measurement_drift"] = _measurement_drift(casedir)
    return result


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


def _write_channel_ic(casedir, setup, params, case, n_cells):
    """Overwrite 0/U with the case's analytic profile at the cell centres.

    blockMesh orders cells x-fastest, so each y-row's value repeats
    _N_STREAMWISE times consecutively.
    """
    from betaflow.analytic import casson as _casson

    h = float(case["geometry"]["half_height"])
    dy = 2.0 * h / n_cells
    y = -h + dy * (np.arange(n_cells) + 0.5)
    u0 = _casson.velocity(y, params["tau0"] / (setup["meta"]["xi"] * h), params["tau0"],
                          params["nu_c"], h)
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


def _final_ux_residual(casedir):
    """Final initial-residual of the Ux equation, parsed from the solver log.

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
    return float(residuals[-1])


def _ux_residual_history(casedir):
    log = (casedir / "log.foamRun").read_text()
    return [
        float(x)
        for x in re.findall(r"Solving for Ux, Initial residual = ([0-9eE.+-]+)", log)
    ]


def _iteration_count(casedir):
    return len(_ux_residual_history(casedir))


def _iterations_to_residual(casedir, tol):
    """First iteration whose Ux residual falls below tol, or None.

    Reported as a comparable stiffness measure across constitutive models: a
    geometrically-contracting fixed point reaches a given tolerance in a
    count that scales with the mesh only, while an algebraically-contracting
    one (casson) blows up with the model's own stiffness parameter.
    """
    for i, value in enumerate(_ux_residual_history(casedir), start=1):
        if value < tol:
            return i
    return None


def _measurement_drift(casedir):
    """Relative change in the sampled profile between the last two written
    states — a measurement-directed convergence criterion for the Casson
    fixed point, whose residual decays too slowly to gate on."""
    sample_root = casedir / "postProcessing" / "sample"
    times = sorted(
        (d for d in sample_root.iterdir() if d.is_dir()), key=lambda d: float(d.name)
    )
    if len(times) < 2:
        raise RuntimeError(f"need two sampled states for a drift check in {casedir}")
    late = np.loadtxt(times[-1] / "centreline.xy")[:, 1]
    early = np.loadtxt(times[-2] / "centreline.xy")[:, 1]
    return float(np.max(np.abs(late - early)) / np.max(np.abs(late)))


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


def _read_tau_wall(casedir, patches=("bottomWall", "topWall")):
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
    for patch in patches:
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


def _read_nu_eff(casedir):
    """Effective viscosity at the sampled stations, from the solver's own field.

    The sets FO writes one file per field; nuEff is a scalar so its file has
    columns y, nuEff.
    """
    sample_root = casedir / "postProcessing" / "sample"
    times = sorted(
        (d for d in sample_root.iterdir() if d.is_dir()), key=lambda d: float(d.name)
    )
    data = np.loadtxt(times[-1] / "centreline.xy")
    # The sets writer emits ONE file per set with every requested field
    # side by side: y, U_x, U_y, U_z, then the scalar viscosity.
    if data.shape[1] < 5:
        raise RuntimeError(
            f"expected 5 columns (y, U, nu) in {times[-1]}/centreline.xy, "
            f"got {data.shape[1]} — was the viscosity field requested?"
        )
    return data[:, 4]


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

_WOMERSLEY_TRANSPORT = {
    "womersley": "momentumTransport",
    "womersley_carreau": "momentumTransport_carreau",
    "pipe_womersley": "momentumTransport",
}

# HARDCODING FOUND (pipe port): the transient file set assumed a box mesh, a
# box 0/p, and a two-wall functions dict.
_WOMERSLEY_GEOMETRY_FILES = {
    "womersley": {"blockMeshDict": "system/blockMeshDict", "p": "0/p",
                  "functions_womersley": "system/functions"},
    "womersley_carreau": {"blockMeshDict": "system/blockMeshDict", "p": "0/p",
                          "functions_womersley": "system/functions"},
    "pipe_womersley": {"blockMeshDict_wedge": "system/blockMeshDict",
                       "p_pipe": "0/p",
                       "functions_womersley_pipe": "system/functions"},
}

_WOMERSLEY_FILES = {
    "blockMeshDict": "system/blockMeshDict",
    "controlDict_womersley": "system/controlDict",
    "fvSchemes_womersley": "system/fvSchemes",
    "fvSolution_womersley": "system/fvSolution",
    "fvModels_womersley": "constant/fvModels",
    "physicalProperties": "constant/physicalProperties",
}

_N_PHASES = 16  # profile writes per cycle (periodicity check + harmonic fit)

_DDT_SCHEMES = {"backward": "backward", "Euler": "Euler"}


def _run_womersley(
    case,
    n_cells=None,
    alpha=None,
    cu=None,
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
    gtype_early = geom["type"]
    # h is the transverse half-extent: half-height for a channel, RADIUS for
    # a pipe. alpha is defined on it in both cases (Womersley 1955 uses the
    # pipe radius), so the formula is shared but the symbol is not.
    h = float(geom["radius"] if gtype_early == "pipe_womersley" else geom["half_height"])
    length = float(geom["length"])
    alpha = float(alpha if alpha is not None else case["nondim"]["alpha"])
    res = case["resolution"]
    n_steps = int(n_steps if n_steps is not None else res["n_steps_per_cycle"])
    conv = case["convergence"]
    max_cycles = int(max_cycles if max_cycles is not None else conv["max_cycles"])
    periodicity_tol = float(conv["periodicity_tol"])

    gtype = geom["type"]
    period = 1.0  # s; with power-of-two n_steps all write times are binary-exact
    omega = 2.0 * np.pi / period
    # alpha = h sqrt(omega/nu) => nu = omega h^2 / alpha^2 — must match
    # betaflow/analytic/womersley.py (half-height h). For the Carreau variant
    # this nu is nu0, so alpha here is the NOMINAL Womersley number.
    nu = omega * h**2 / alpha**2

    rheology = {}
    if gtype == "womersley_carreau":
        from betaflow.analytic import carreau as _carreau

        nd = case["nondim"]
        cu = float(nd["Cu"] if cu is None else cu)
        n_index = float(nd["n"])
        a_index = float(nd.get("a", 2.0))
        nu_inf = float(nd["nu_inf_over_nu0"]) * nu
        # Cu = k G h / nu0 with G PRESCRIBED here (unlike the steady case,
        # where the meanVelocityForce constraint made G an output), so k
        # follows directly — no coupled solve.
        k = cu * nu / (g_amp * h)
        # Size the mesh on the WALL viscosity, not nu0: shear thinning makes
        # the Stokes layer THINNER and the effective Womersley number HIGHER
        # than nominal. Estimated quasi-steadily at the peak wall stress G h.
        gdot_wall = _carreau.shear_rate(g_amp * h, nu, nu_inf, k, n_index, a_index)
        nu_wall = float(_carreau.viscosity(gdot_wall, nu, nu_inf, k, n_index, a_index))
        rheology = {
            "nu0": nu,
            "nu_inf": nu_inf,
            "k": k,
            "n": n_index,
            "a": a_index,
            "Cu": cu,
            "nu_wall_estimate": nu_wall,
            "alpha_nominal": alpha,
            "alpha_eff": float(h * np.sqrt(omega / nu_wall)),
        }
        nu_sizing = nu_wall
    else:
        nu_sizing = nu

    delta = np.sqrt(2.0 * nu_sizing / omega)
    if n_cells is None:
        # Mesh level = cells per Stokes layer, rounded to a multiple of 8.
        # HARDCODING FOUND (pipe port): this used 2*h unconditionally, i.e. it
        # assumed the mesh spans the FULL transverse extent as a channel does.
        # A pipe is meshed over the radius only, so the same formula
        # over-resolved it 2x — which then pushed the viscous Fourier number
        # to 6.2 where the channel runs at 1.5, and the transient diverged.
        # A sizing bug presenting as a stability failure.
        transverse = h if gtype_early == "pipe_womersley" else 2.0 * h
        cpd = float(res["cells_per_stokes_layer"])
        n_cells = int(round(cpd * transverse / delta / 8.0)) * 8
    n_cells = int(n_cells)
    u_ref = g_amp / omega
    # Lever arm in the momentum balance: tau_w = LEVER*(G - d<u>/dt). It is h
    # for a channel (volume/area = h) and a/2 for a pipe. This was hardcoded
    # to h; the identity caught it instantly, reading exactly 1.0 — a 100%
    # error — because the two differ by precisely the factor of two that
    # tau(r) = G r/2 introduces.
    lever = 0.5 * h if geom["type"] == "pipe_womersley" else h
    tau_ref = g_amp * lever

    workdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    tag = f"_cu{rheology['Cu']:g}" if rheology else ""
    casedir = (
        workdir / f"{case['name']}_openfoam_a{alpha:g}{tag}_n{n_cells}_nt{n_steps}_{ddt}"
    )
    if casedir.exists():
        shutil.rmtree(casedir)

    is_pipe = gtype == "pipe_womersley"
    transverse_extent = h if is_pipe else 2.0 * h
    gap = transverse_extent
    eps = 5e-7 * gap
    params = {
        "length": length,
        "y_min": 0.0 if is_pipe else -h,
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
        "y_start": (0.0 if is_pipe else -h) + eps,
        "y_end": h - eps,
        # The nonlinear viscosity must be converged WITHIN each timestep, or
        # the lagged nu injects an O(dt) error into exactly the quantities
        # this case measures. Newtonian needs no outer loop.
        "n_outer": 3 if rheology else 1,
        "nu_inf": rheology.get("nu_inf", 0.0),
        "k": rheology.get("k", 0.0),
        "n": rheology.get("n", 1.0),
        "a": rheology.get("a", 2.0),
        **(_wedge_params(h, n_cells) if is_pipe else {}),
    }
    files = {**_WOMERSLEY_FILES, **_WOMERSLEY_GEOMETRY_FILES[gtype]}
    files[_WOMERSLEY_TRANSPORT[gtype]] = "constant/momentumTransport"
    for src, dest in files.items():
        target = casedir / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(Template((TEMPLATE_DIR / src).read_text()).substitute(params))
    # Initial condition: the linear Womersley profile at the EFFECTIVE alpha.
    # Exact for the Newtonian case; for Carreau it is only a good guess, which
    # is all an initial condition has to be — periodicity is measured, not
    # assumed.
    if is_pipe:
        from betaflow.analytic import pipe as _pipe

        _write_pipe_womersley_ic(casedir, h, n_cells, alpha, u_ref, _pipe)
    else:
        _write_womersley_ic(
            casedir, h, n_cells, rheology.get("alpha_eff", alpha), u_ref, _wom
        )

    _foam(casedir, "blockMesh")
    _foam(casedir, "foamRun")

    profiles = _read_phase_profiles(casedir)
    ratios, cycles_to_periodic = _periodicity(profiles, period, max_cycles, periodicity_tol)
    y, u_hat = _harmonic_profile(profiles, period, max_cycles)
    _, u_hat3 = _harmonic_profile(profiles, period, max_cycles, harmonic=3)
    half_wave = _half_wave_residual(profiles, period, max_cycles)

    t_series, tau_b, tau_t, u_bulk = _read_wall_series(casedir, single_wall=is_pipe)
    identity_max, sign_b, sign_t, tau_series = _wall_identity(
        t_series, tau_b, tau_t, u_bulk, lever, g_amp, omega, ddt, period / n_steps
    )
    # The momentum-balance identity is the convergence gate for the transient
    # generalised-Newtonian case, exactly as it is for the steady ones — and
    # here it is also the ONLY exact pointwise check available. A run can
    # complete without crashing yet be numerically worthless: the variable-nu
    # explicit deviatoric term seeds a spurious wall-normal velocity, the two
    # wall shears diverge from each other, and the identity opens to O(1)
    # while the solver reports nothing. Refuse such a run rather than report
    # its profile.
    if gtype == "womersley_carreau" and identity_max > _IDENTITY_GATE_TOL:
        fourier = nu * (period / n_steps) / (transverse_extent / n_cells) ** 2
        raise RuntimeError(
            f"momentum-balance identity open by {identity_max:.3e} > "
            f"{_IDENTITY_GATE_TOL:.0e} in {casedir}. The viscous Fourier "
            f"number nu0*dt/dy^2 is {fourier:.1f}; the variable-viscosity "
            f"explicit term is stable only for O(1) values, so n_steps must "
            f"scale as n_cells^2. See log.foamRun."
        )
    tau_hat = _harmonic_scalar(t_series, tau_series, period, max_cycles)
    tau_hat3 = _harmonic_scalar(t_series, tau_series, period, max_cycles, harmonic=3)

    return {
        "y": y,
        "u_amp": np.abs(u_hat),
        "u_phase": np.angle(u_hat),
        "u_ref": u_ref,
        "tau_amp": float(np.abs(tau_hat)),
        "tau_phase": float(np.angle(tau_hat)),
        "u_amp3": np.abs(u_hat3),
        "tau_amp3": float(np.abs(tau_hat3)),
        "tau_ref": tau_ref,
        "meta": {
            "solver": "openfoam",
            "openfoam_version": _openfoam_version(),
            "mesh_level": n_cells,
            "n_cells_total": _N_STREAMWISE * n_cells,
            "nu": nu,
            "alpha": alpha,
            "cells_per_stokes_layer": n_cells * delta / transverse_extent,
            **rheology,
            "n_steps_per_cycle": n_steps,
            "ddt": ddt,
            "cycles_run": max_cycles,
            "cycles_to_periodic": cycles_to_periodic,
            "periodicity": ratios,
            "identity_max_rel": identity_max,
            "half_wave_residual": half_wave,
            "viscous_fourier": float(
                nu * (period / n_steps) / (transverse_extent / n_cells) ** 2
            ),
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


def _write_pipe_womersley_ic(casedir, a, n_cells, alpha, u_ref, pipemod):
    """0/U with the analytic J0 Womersley profile at t = 0, at cell centres.

    Wedge cell centroids are area-weighted in r, not at (i+1/2)dr: for an
    annular sector between r1 and r2 the centroid sits at
    (2/3)(r2^3-r1^3)/(r2^2-r1^2). Using the naive midpoint here would seed an
    O(dr) error into the initial condition.
    """
    edges = np.linspace(0.0, a, n_cells + 1)
    r1, r2 = edges[:-1], edges[1:]
    r_c = (2.0 / 3.0) * (r2**3 - r1**3) / (r2**2 - r1**2)
    u0 = u_ref * np.real(pipemod.womersley_profile(r_c / a, alpha))
    values = np.repeat(u0, _N_STREAMWISE)
    entries = "\n".join(f"({v:.12g} 0 0)" for v in values)
    (casedir / "0" / "U").write_text(
        "FoamFile\n{\n    format      ascii;\n    class       volVectorField;\n"
        "    object      U;\n}\n\n"
        "dimensions      [0 1 -1 0 0 0 0];\n\n"
        f"internalField   nonuniform List<vector>\n{values.size}\n(\n{entries}\n);\n\n"
        "boundaryField\n{\n"
        "    inlet  { type cyclic; }\n"
        "    outlet { type cyclic; }\n"
        "    wall   { type noSlip; }\n"
        "    front  { type wedge; }\n"
        "    back   { type wedge; }\n"
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


def _half_wave_residual(profiles, period, max_cycles):
    """max ||u(t+T/2) + u(t)|| / max ||u|| over the final cycle.

    G(t) = G cos(wt) is odd under t -> t + T/2 and nu(|gammadot|) is even in
    gammadot, so the periodic state MUST satisfy u(y, t+T/2) = -u(y, t)
    exactly. Any violation is a bug or incomplete periodicity, never physics
    — it holds for a nonlinear rheology just as it does for a linear one.
    """
    half = _N_PHASES // 2
    diffs, norms = [], []
    for m in range(1, half + 1):
        t = (max_cycles - 1) * period + m * period / _N_PHASES
        _, u_a = _profile_at(profiles, t)
        _, u_b = _profile_at(profiles, t + period / 2.0)
        diffs.append(np.linalg.norm(u_a + u_b))
        norms.append(np.linalg.norm(u_a))
    return float(max(diffs) / max(norms))


def _harmonic_profile(profiles, period, max_cycles, harmonic=1):
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
        term = u * np.exp(-2.0j * np.pi * harmonic * t / period)
        acc = term if acc is None else acc + term
    return y, 2.0 / _N_PHASES * acc


def _harmonic_scalar(t, series, period, max_cycles, harmonic=1):
    """Complex amplitude of one harmonic of a per-timestep scalar series,
    over the final cycle. Half-wave symmetry admits ONLY odd harmonics, and
    a linear (Newtonian) response has none above the first — so the third
    harmonic is a direct measure of the constitutive nonlinearity."""
    mask = t > (max_cycles - 1) * period + 1e-9
    tt, ss = t[mask], series[mask]
    return 2.0 / tt.size * np.sum(ss * np.exp(-2.0j * np.pi * harmonic * tt / period))


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


def _read_wall_series(casedir, single_wall=False):
    """Per-timestep x-components: wall shear on each wall and bulk velocity.

    A pipe has ONE wall patch, so its single series stands in for both and the
    two-wall symmetry check below is vacuously satisfied.
    """
    root = casedir / "postProcessing"
    if single_wall:
        t_b, tau_b = _read_vector_series(root / "tauWall" / "0" / "surfaceFieldValue.dat")
        t_t, tau_t = t_b, tau_b
    else:
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
