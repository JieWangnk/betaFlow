"""OpenFOAM 14 (Foundation) particle adapter: the brownianTracer cloud.

Runs betaflow's particle cases through OpenFOAM's NEW modular Lagrangian
framework using the custom `brownianTracer` cloud — a displacement-level
Brownian walk (position-space Euler-Maruyama) written after the stock
`BrownianMotionForce` was measured to deliver a maxCo-dependent diffusivity
(D/D_SE = 0.38-0.59 on the development machine, 2026-08-08). The cloud's
construction invariant: noise is a per-global-step velocity, and the
framework's tracking only subdivides the one displacement deltaT*U, so the
amplitude interval and the applied interval are the same deltaT always.

Requires `libbrownianTracerCloud.so` in $FOAM_USER_LIBBIN, built from
~/OpenFOAM/of14-catchup/diffusiveTracer/src/brownianTracerCloud (wmake libso
under OpenFOAM 14). The runner fails early with a clear message otherwise.

Cases served (branch on the presence of a `flow:` key, as runners/langevin.py):

  langevin_free — N particles at the origin of a quiescent 40 um box;
      returns per-time ensemble MSD. Walls are unreachable by construction.
  taylor_aris   — N particles seeded area-uniform (equilibrium P(r) = 2r/a^2)
      in a frozen analytic Poiseuille pipe (solver `functions` +
      `subSolver incompressibleFluid`: no flow solve, tracking cost only);
      returns per-write axial variance, final r/a sample, and the KS gate.
      Walls are specular (`brownianReboundVelocity`: reflects the noise
      velocity as well as U, else recalculation un-reflects it and the wall
      is effectively sticky — measured KS 3.31x floor, D_eff +8.3%).

Conventions shared with runners/langevin.py (load-bearing):
  dt      = epsilon**2 * tau_r / 2
  n_steps = int(round(2 * cycles / epsilon**2))
  D_expected     = brownian.stokes_einstein(T, mu, a_p)
  D_eff_expected = taylor_aris.d_eff(D, a, U)
The tests read the exact values from this dict, never from case["reference"].

Mesh notes (both bit the development runs): the O-grid pipe rim is faceted,
so seeds are capped at r <= a*cos(pi/n_facets) (48 facets -> 0.43% CDF
truncation, a third of the KS floor at N = 1e4); seeding outside the facets
is silently rejected by injection.
"""
from __future__ import annotations

import os
import re
import shutil
import sys
from pathlib import Path
from string import Template

import numpy as np

from betaflow.analytic import brownian
from betaflow.analytic import taylor_aris as ta
from betaflow.runners.openfoam import _bashrc, _foam, _openfoam_version

# * * * * * * * * * * * * * * * * templates * * * * * * * * * * * * * * * * #
# Solver-specific dictionaries stay inside this module (layering rule:
# nothing above runners/ may know OpenFOAM exists).

_HEADER = """FoamFile
{
    format      ascii;
    class       $cls;
    location    "$loc";
    object      $obj;
}
"""


def _dict(cls, loc, obj, body):
    return Template(_HEADER).substitute(cls=cls, loc=loc, obj=obj) + body


_CONTROL = """
application     foamRun;
solver          $solver;
$subsolver
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         $end_time;
deltaT          $delta_t;
adjustTimeStep  no;

writeControl    timeStep;
writeInterval   $write_interval;
purgeWrite      0;
writeFormat     ascii;
writePrecision  12;
writeCompression off;
timeFormat      general;
timePrecision   10;
runTimeModifiable no;
"""

_FUNCTIONS = """
cloud
{
    type            brownianTracerCloud;
    libs            ("libbrownianTracerCloud.so");
    executeControl  timeStep;
    executeInterval 1;
    writeControl    writeTime;

    diffusivity     $diffusivity;
    seed            $seed;
}

position
{
    type            cloudPosition;
    libs            ("libLagrangianCloudFunctionObjects.so");
    cloud           cloud;
}
"""

_LAG_SCHEMES = """
tracking    linear;

ddtSchemes           { default Euler; }
SpSchemes            { default none; }
averagingSchemes     {}
interpolationSchemes
{
    default     cell;
    Uc          cellPoint;
}
accumulationSchemes  {}
"""

_LAG_SOLUTION = """
maxTimeStepFraction         0.3;
maxCellLengthScaleFraction  0.3;
nCorrectors                 0;
"""

_LAG_MODELS = """
inlet
{
    type        manualInjection;
    file        "injectionPositions";
    time        0 [s];
}
"""

_BOX_BLOCKMESH = """
convertToMeters 1e-6;

vertices
(
    (-20 -20 -20) ( 20 -20 -20) ( 20  20 -20) (-20  20 -20)
    (-20 -20  20) ( 20 -20  20) ( 20  20  20) (-20  20  20)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) (20 20 20) simpleGrading (1 1 1)
);

boundary
(
    walls
    {
        type wall;
        faces
        (
            (0 3 2 1) (4 5 6 7)
            (0 1 5 4) (2 3 7 6)
            (0 4 7 3) (1 2 6 5)
        );
    }
);
"""

_FV_SCHEMES = """
ddtSchemes           { default Euler; }
gradSchemes          { default Gauss linear; }
divSchemes
{
    default         none;
    div(phi,U)      Gauss upwind;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes     { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes        { default corrected; }
"""

_FV_SOLUTION = """
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-7;
        relTol          0.1;
        smoother        DICGaussSeidel;
    }
    pFinal   { $$p; relTol 0; }
    U
    {
        solver          smoothSolver;
        smoother        GaussSeidel;
        tolerance       1e-8;
        relTol          0;
    }
    UFinal   { $$U; }
}

PIMPLE
{
    momentumPredictor   no;
    nOuterCorrectors    1;
    nCorrectors         1;
    nNonOrthogonalCorrectors 0;
    pRefCell            0;
    pRefValue           0;
}
"""

_PHYS = """
viscosityModel  constant;
nu              [0 2 -1 0 0 0 0] 1e-6;
"""

_MOM = """
simulationType  laminar;
"""

_N_FACETS = 48          # ogrid outer facets at n_c = 12 (4 * n_c)


# * * * * * * * * * * * * * * * * helpers * * * * * * * * * * * * * * * * * #

def _require_library():
    # Always resolve through the bashrc the runs will use: the ambient shell
    # may have a DIFFERENT OpenFOAM sourced (fresh terminals on this machine
    # default to OF12, whose FOAM_USER_LIBBIN never holds this library)
    import subprocess
    libbin = subprocess.check_output(
        ["bash", "-c",
         f"source {_bashrc()} 2>/dev/null && echo -n $FOAM_USER_LIBBIN"],
        text=True).strip()
    lib = Path(libbin) / "libbrownianTracerCloud.so"
    if not lib.is_file():
        raise RuntimeError(
            f"libbrownianTracerCloud.so not found at {lib}. Build it with "
            "wmake libso under OpenFOAM 14 from "
            "~/OpenFOAM/of14-catchup/diffusiveTracer/src/brownianTracerCloud"
        )
    return lib


def _casedir(name, workdir):
    workdir = Path(workdir) if workdir is not None else Path.cwd() / "_runs"
    casedir = workdir / name
    if casedir.exists():
        shutil.rmtree(casedir)
    for sub in ("0", "constant/Lagrangian/cloud", "system/Lagrangian/cloud"):
        (casedir / sub).mkdir(parents=True, exist_ok=True)
    return casedir


def _write(casedir, rel, text):
    (casedir / rel).write_text(text)


def _write_cloud_dicts(casedir, diffusivity, seed):
    _write(casedir, "system/functions",
           _dict("dictionary", "system", "functions",
                 Template(_FUNCTIONS).substitute(
                     diffusivity=repr(diffusivity), seed=seed)))
    _write(casedir, "system/Lagrangian/cloud/LagrangianSchemes",
           _dict("dictionary", "system/Lagrangian/cloud", "LagrangianSchemes",
                 _LAG_SCHEMES))
    _write(casedir, "system/Lagrangian/cloud/LagrangianSolution",
           _dict("dictionary", "system/Lagrangian/cloud", "LagrangianSolution",
                 _LAG_SOLUTION))
    _write(casedir, "constant/Lagrangian/cloud/LagrangianModels",
           _dict("dictionary", "constant/Lagrangian/cloud", "LagrangianModels",
                 _LAG_MODELS))


def _write_positions(casedir, pos):
    lines = [_dict("vectorField", "constant/Lagrangian/cloud",
                   "injectionPositions", "(\n")]
    for p in pos:
        lines.append(f"({p[0]:.10e} {p[1]:.10e} {p[2]:.10e})\n")
    lines.append(")\n")
    _write(casedir, "constant/Lagrangian/cloud/injectionPositions",
           "".join(lines))


def _read_positions(casedir, tdir):
    p = casedir / tdir / "Lagrangian" / "cloud" / "position"
    if not p.is_file():
        return None
    txt = p.read_text()
    m = re.search(r"nonuniform\s+List<vector>\s*\n?\s*(\d+)\s*\n\(", txt)
    if not m:
        return None
    body = txt[m.end():]
    body = body[:body.find("\n)")]
    return np.array([[float(x) for x in s.split()]
                     for s in re.findall(r"\(([^)]*)\)", body)])


def _time_dirs(casedir):
    out = []
    for d in os.listdir(casedir):
        if not (casedir / d).is_dir():
            continue
        try:
            out.append((float(d), d))
        except ValueError:
            continue
    return sorted(out)


def _ks_statistic(r_over_a):
    """One-sample KS against F(r) = (r/a)^2, no scipy needed."""
    x = np.sort(np.clip(np.asarray(r_over_a, dtype=float), 0.0, 1.0))
    n = len(x)
    cdf = x**2
    lo = np.max(np.abs(cdf - np.arange(0, n) / n))
    hi = np.max(np.abs(cdf - np.arange(1, n + 1) / n))
    return float(max(lo, hi))


def _meta_common(casedir, seed, dt, n_steps, n_particles, diffusivity):
    lib = _require_library()
    return {
        "solver": "openfoam_particles",
        "openfoam_version": _openfoam_version(),
        "cloud": "brownianTracer",
        "cloud_library": str(lib),
        "cloud_library_mtime": int(lib.stat().st_mtime),
        "seed": int(seed),
        "dt": float(dt),
        "n_steps": int(n_steps),
        "n_particles": int(n_particles),
        "diffusivity": float(diffusivity),
        "case_dir": str(casedir),
    }


# * * * * * * * * * * * * * * langevin_free branch * * * * * * * * * * * * * #

def _run_free(case, n_particles, seed, n_steps, total_time, workdir):
    phys = case["physical"]
    a_p = float(phys["particle_radius"])
    T = float(phys["temperature"])
    mu = float(phys["dynamic_viscosity"])
    rho_p = float(phys["particle_density"])
    if total_time is None:
        total_time = float(phys["total_time"])
    dt = total_time / n_steps

    D = brownian.stokes_einstein(T, mu, a_p)

    casedir = _casedir(
        f"{case['name']}_openfoam_particles_n{n_particles}_s{seed}", workdir)

    _write(casedir, "system/blockMeshDict",
           _dict("dictionary", "system", "blockMeshDict", _BOX_BLOCKMESH))
    _write(casedir, "system/controlDict",
           _dict("dictionary", "system", "controlDict",
                 Template(_CONTROL).substitute(
                     solver="incompressibleFluid", subsolver="",
                     end_time=repr(total_time), delta_t=repr(dt),
                     write_interval=1)))
    _write(casedir, "system/fvSchemes",
           _dict("dictionary", "system", "fvSchemes", _FV_SCHEMES))
    _write(casedir, "system/fvSolution",
           _dict("dictionary", "system", "fvSolution",
                 Template(_FV_SOLUTION).substitute()))
    _write(casedir, "constant/physicalProperties",
           _dict("dictionary", "constant", "physicalProperties", _PHYS))
    _write(casedir, "constant/momentumTransport",
           _dict("dictionary", "constant", "momentumTransport", _MOM))
    _write(casedir, "0/U", _dict("volVectorField", "0", "U", """
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    walls   { type noSlip; }
}
"""))
    _write(casedir, "0/p", _dict("volScalarField", "0", "p", """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    walls   { type zeroGradient; }
}
"""))
    _write_cloud_dicts(casedir, D, seed)
    _write_positions(casedir, np.zeros((n_particles, 3)))

    _foam(casedir, "blockMesh")
    _foam(casedir, "foamRun")

    ts, msd, comps = [0.0], [0.0], [np.zeros(3)]
    for t, d in _time_dirs(casedir):
        if t <= 0:
            continue
        pts = _read_positions(casedir, d)
        if pts is None or len(pts) == 0:
            continue
        ts.append(t)
        msd.append(float(np.mean(np.sum(pts**2, axis=1))))
        comps.append(np.mean(pts**2, axis=0))
    t = np.array(ts)
    msd = np.array(msd)
    msd_components = np.vstack(comps)

    meta = _meta_common(casedir, seed, dt, n_steps, n_particles, D)
    meta.update({
        "total_time": float(total_time),
        "friction_coefficient": brownian.friction_coefficient(mu, a_p),
        "stokes_number": brownian.stokes_number(rho_p, a_p, mu, 0.3, 0.01),
        "rms_displacement": float(np.sqrt(msd[-1])),
    })
    return {
        "t": t,
        "msd": msd,
        "msd_components": msd_components,
        "D_expected": D,
        "meta": meta,
    }


# * * * * * * * * * * * * shared pipe construction * * * * * * * * * * * * #

def _write_pipe_case(casedir, a, length, end_time, dt, write_every, u_mean):
    """O-grid pipe, cyclic axial BCs, frozen analytic Poiseuille velocity.

    Extracted verbatim from the taylor_aris branch so the CIR branch shares
    one code path. The velocity written at the cell centres is the exact
    parabola; what a PARTICLE then samples is the cellPoint interpolant,
    whose bias is directional (see _run_cir).
    """
    tools = Path(__file__).resolve().parents[2] / "tools"
    argv = sys.argv
    try:
        sys.argv = ["ogrid_blockmesh.py", str(casedir), "1", "1"]
        sys.path.insert(0, str(tools))
        import ogrid_blockmesh as og
    finally:
        sys.argv = argv
        sys.path.remove(str(tools))
    nx = int(np.ceil(length / 20e-6))
    _write(casedir, "system/blockMeshDict", og.ogrid_dict(a, length, 12, 8, nx))

    _write(casedir, "system/controlDict",
           _dict("dictionary", "system", "controlDict",
                 Template(_CONTROL).substitute(
                     solver="functions",
                     subsolver="subSolver       incompressibleFluid;",
                     end_time=repr(end_time), delta_t=repr(dt),
                     write_interval=write_every)))
    _write(casedir, "system/fvSchemes",
           _dict("dictionary", "system", "fvSchemes", _FV_SCHEMES))
    _write(casedir, "system/fvSolution",
           _dict("dictionary", "system", "fvSolution",
                 Template(_FV_SOLUTION).substitute()))
    _write(casedir, "constant/physicalProperties",
           _dict("dictionary", "constant", "physicalProperties", _PHYS))
    _write(casedir, "constant/momentumTransport",
           _dict("dictionary", "constant", "momentumTransport", _MOM))

    cyc = ("boundaryField\n{\n    inlet   { type cyclic; }\n"
           "    outlet  { type cyclic; }\n")
    _write(casedir, "0/p", _dict("volScalarField", "0", "p",
           "\ndimensions      [0 2 -2 0 0 0 0];\ninternalField   uniform 0;\n"
           + cyc + "    wall    { type zeroGradient; }\n}\n"))
    _write(casedir, "0/U", _dict("volVectorField", "0", "U",
           "\ndimensions      [0 1 -1 0 0 0 0];\n"
           "internalField   uniform (0 0 0);\n"
           + cyc + "    wall    { type noSlip; }\n}\n"))

    _foam(casedir, "blockMesh")
    _foam(casedir, "foamPostProcess -func writeCellCentres -time 0")

    txt = (casedir / "0" / "C").read_text()
    m = re.search(r"internalField\s+nonuniform\s+List<vector>\s*\n?\s*(\d+)\s*\n\(",
                  txt)
    n = int(m.group(1))
    body = txt[m.end():]
    body = body[:body.find("\n)")]
    C = np.array([[float(x) for x in s.split()]
                  for s in re.findall(r"\(([^)]*)\)", body)])
    r2 = (C[:, 1]**2 + C[:, 2]**2) / a**2
    u = 2.0 * u_mean * np.clip(1.0 - r2, 0.0, None)
    with open(casedir / "0" / "U", "w") as f:
        f.write(_dict("volVectorField", "0", "U",
                      "\ndimensions      [0 1 -1 0 0 0 0];\n"
                      "internalField   nonuniform List<vector>\n")
                + f"{n}\n(\n")
        for ui in u:
            f.write(f"({ui:.10e} 0 0)\n")
        f.write(")\n;\n\n" + cyc
                + "    wall    { type fixedValue; value uniform (0 0 0); }\n}\n")


def _seed_area_uniform(n_particles, a, x0, seed):
    """Equilibrium P(r) = 2r/a^2 inside the faceted rim, delta pulse at x0."""
    rng = np.random.default_rng(seed)
    cap = np.cos(np.pi / _N_FACETS)**2
    r = a * np.sqrt(rng.random(n_particles) * cap)
    th = rng.random(n_particles) * 2 * np.pi
    pos = np.column_stack([np.full(n_particles, x0),
                           r * np.cos(th), r * np.sin(th)])
    return pos, cap


# * * * * * * * * * * * * * * taylor_aris branch * * * * * * * * * * * * * * #

def _run_pipe(case, n_particles, seed, epsilon, cycles, particle_radius,
              workdir, writes):
    phys = case["physical"]
    a = float(phys["vessel_radius"])
    u_mean = float(phys["mean_velocity"])
    T = float(phys["temperature"])
    mu = float(phys["dynamic_viscosity"])
    rho_p = float(phys["particle_density"])
    r_p = particle_radius if particle_radius is not None \
        else float(phys["particle_radius"])

    D = brownian.stokes_einstein(T, mu, r_p)
    tau_r = ta.radial_relaxation_time(a, D)
    # load-bearing: identical to runners/langevin.py
    dt = epsilon**2 * tau_r / 2.0
    n_steps = int(round(2.0 * cycles / epsilon**2))
    end_time = n_steps * dt
    write_every = max(1, n_steps // writes)

    # domain long enough that no particle reaches either axial end (4 sigma)
    drift = 2.0 * u_mean * end_time            # centreline speed bound
    spread = 4.0 * np.sqrt(2.0 * ta.d_eff(D, a, u_mean) * end_time)
    x0 = 0.1 * (drift + spread)
    length = x0 + drift + spread

    casedir = _casedir(
        f"{case['name']}_openfoam_particles_n{n_particles}_s{seed}"
        f"_eps{epsilon:g}_c{cycles:g}", workdir)

    _write_pipe_case(casedir, a, length, end_time, dt, write_every, u_mean)
    _write_cloud_dicts(casedir, D, seed)
    pos, cap = _seed_area_uniform(n_particles, a, x0, seed)
    _write_positions(casedir, pos)

    _foam(casedir, "foamRun")

    ts, var_x, ks = [], [], []
    r_final = None
    for t, d in _time_dirs(casedir):
        pts = _read_positions(casedir, d)
        if pts is None or len(pts) == 0:
            continue
        ts.append(t)
        var_x.append(float(np.var(pts[:, 0])))
        r_over_a = np.hypot(pts[:, 1], pts[:, 2]) / a
        ks.append(_ks_statistic(r_over_a))
        r_final = r_over_a
    t = np.array(ts)
    var_x = np.array(var_x)
    ks = np.array(ks)

    # ~8 evenly spaced KS checkpoints, as the reference runner reports
    idx = np.unique(np.linspace(0, len(t) - 1, 8).astype(int))
    ks_history = {
        "t_over_tau_r": [float(t[i] / tau_r) for i in idx],
        "statistic": [float(ks[i]) for i in idx],
    }

    meta = _meta_common(casedir, seed, dt, n_steps, n_particles, D)
    meta.update({
        "tau_r": float(tau_r),
        "particle_radius": float(r_p),
        "peclet": float(u_mean * a / D),
        "stokes_number": brownian.stokes_number(rho_p, r_p, mu, 0.3, 0.01),
        "wall_scheme": "specular rebound (brownianReboundVelocity: reflects "
                       "U and the noise velocity; e=1, mu=0)",
        "epsilon_radial_step": float(epsilon),
        "cycles_tau_r": float(cycles),
        "pipe_length": float(length),
        "seed_x0": float(x0),
        "rim_cap_r_over_a": float(np.sqrt(cap)),
    })
    return {
        "t": t,
        "var_x": var_x,
        "r_over_a": r_final,
        "ks_statistic": float(ks[-1]),
        "ks_history": ks_history,
        "D_expected": D,
        "D_eff_expected": ta.d_eff(D, a, u_mean),
        "meta": meta,
    }


# * * * * * * * * * * * * * * mc_channel branch * * * * * * * * * * * * * * #

def _run_cir(case, n_particles, seed, workdir, writes, epsilon,
             diffusivity=None, time_horizon_over_t2=None):
    """Channel impulse response through the OpenFOAM Lagrangian tracker.

    The method-class replication of Hofmann et al. 2024: one-way-coupled
    Lagrangian particles in frozen Poiseuille flow, uniform-area release,
    transparent axial receiver windows. Their published model has no
    diffusion, so the D = 0 leg IS the replication of their model class;
    D > 0 exceeds it with the validated Brownian walk. (Their exact MPPIC
    setup is not reproducible: stock OF14 registers the Brownian force only
    for thermo-family clouds MPPIC cannot construct, and their DMPPIC source
    is deleted from GitHub with no archived copy.)

    PREDICTION, WRITTEN BEFORE THE FIRST RUN. The particles sample the
    cellPoint INTERPOLANT of the frozen parabola, and linear interpolation
    of a concave profile underestimates it everywhere. So the OF leg's
    speeds are biased LOW with one sign: arrivals lag the exact solution
    (onset at or after t1, never before), and the whole measured CIR shifts
    late by O((h/a)^2) in time. Unlike the two-flux tail question that was
    called wrong in runners/langevin.py, this mechanism has one flux and
    one sign. The measured shift is recorded either way.

    Axial ends are CYCLIC (the validated configuration); the domain is
    sized so no particle can wrap within end_time, which keeps re-entry
    impossible rather than improbable.
    """
    from betaflow.analytic import channel_impulse as ci_ref

    phys = case["physical"]
    recv = case["receiver"]
    num = case.get("numerics", {})
    a = float(phys["vessel_radius"])
    u_mean = float(phys["mean_velocity"])
    c_x = float(recv["axial_length"])
    distances = [float(d) for d in recv["distances"]]
    D = float(phys.get("diffusivity", 0.0) if diffusivity is None
              else diffusivity)
    horizon = float(num.get("time_horizon_over_t2", 10.0)
                    if time_horizon_over_t2 is None else time_horizon_over_t2)
    resolution = int(num.get("cir_resolution_steps", 50))

    t1 = {d: ci_ref.onset_time(u_mean, d, c_x) for d in distances}
    t2 = {d: ci_ref.peak_time(u_mean, d, c_x) for d in distances}

    # dt resolves the fastest receiver's rise; at D > 0 also the radial walk.
    dt = min(t2.values()) / resolution
    if D > 0.0:
        tau_r = a**2 / D
        dt = min(dt, epsilon**2 * tau_r / 2.0)
    end_time = horizon * max(t2.values())
    n_steps = int(np.ceil(end_time / dt))
    end_time = n_steps * dt
    write_every = max(1, n_steps // writes)

    # No particle may wrap through the cyclic ends within end_time.
    drift = 2.0 * u_mean * end_time
    spread = 4.0 * np.sqrt(2.0 * D * end_time) if D > 0.0 else 0.0
    x0 = 1e-4 + spread
    length = x0 + drift + spread + 2e-4

    casedir = _casedir(
        f"{case['name']}_openfoam_particles_n{n_particles}_s{seed}"
        f"_D{D:g}", workdir)
    _write_pipe_case(casedir, a, length, end_time, dt, write_every, u_mean)
    _write_cloud_dicts(casedir, D, seed)
    pos, cap = _seed_area_uniform(n_particles, a, x0, seed)
    _write_positions(casedir, pos)

    _foam(casedir, "foamRun")

    windows = {d: (x0 + d - c_x / 2.0, x0 + d + c_x / 2.0) for d in distances}
    ts, fractions = [], {d: [] for d in distances}
    r_final = None
    for t, tdir in _time_dirs(casedir):
        if t <= 0:
            continue
        pts = _read_positions(casedir, tdir)
        if pts is None or len(pts) == 0:
            continue
        ts.append(t)
        for d, (lo, hi) in windows.items():
            fractions[d].append(
                float(np.count_nonzero((pts[:, 0] >= lo) & (pts[:, 0] <= hi))
                      / n_particles))
        r_final = np.hypot(pts[:, 1], pts[:, 2]) / a
    tt = np.array(ts)

    receivers = []
    for d in distances:
        rec = {
            "dbar": d,
            "t": tt,
            "cir_measured": np.array(fractions[d]),
            "cir_reference": ci_ref.cir(tt, u_mean, d, c_x),
            "t1": t1[d],
            "t2": t2[d],
            "peak_value": ci_ref.peak_value(d, c_x),
        }
        if D > 0.0:
            rec["flow_dominated_ratio"] = ci_ref.flow_dominated(
                u_mean * a / D, d, a)
        receivers.append(rec)

    meta = _meta_common(casedir, seed, dt, n_steps, n_particles, D)
    meta.update({
        "mode": "cir: uniform release, frozen Poiseuille, transparent receivers",
        "vessel_radius": a,
        "mean_velocity": u_mean,
        "receiver_length": c_x,
        "time_horizon_over_t2": horizon,
        "pipe_length": float(length),
        "seed_x0": float(x0),
        "rim_cap_r_over_a": float(np.sqrt(cap)),
        "interpolation_bias_prediction": (
            "cellPoint interpolation of the concave parabola biases speeds "
            "low; arrivals lag the exact solution, one sign"
        ),
    })
    if D > 0.0:
        meta.update({
            "tau_r": float(a**2 / D),
            "peclet": float(u_mean * a / D),
            "wall_scheme": "specular rebound (brownianReboundVelocity)",
        })
    return {
        "receivers": receivers,
        "r_over_a": r_final,
        "ks_statistic": _ks_statistic(r_final),
        "meta": meta,
    }


# * * * * * * * * * * * * * * * * entry point * * * * * * * * * * * * * * * #

def run(case, n_particles=10000, seed=0, n_steps=100, total_time=None,
        epsilon=0.05, cycles=10.0, particle_radius=None, workdir=None,
        writes=100, diffusivity=None, time_horizon_over_t2=None):
    """Dispatch on the case shape, exactly as runners/langevin.py does."""
    _require_library()
    if "receiver" in case:
        return _run_cir(case, n_particles, seed, workdir, writes, epsilon,
                        diffusivity, time_horizon_over_t2)
    if "flow" in case:
        return _run_pipe(case, n_particles, seed, epsilon, cycles,
                         particle_radius, workdir, writes)
    return _run_free(case, n_particles, seed, n_steps, total_time, workdir)
